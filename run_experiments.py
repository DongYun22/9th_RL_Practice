"""하이퍼파라미터 실험 오케스트레이터.

Stage A(거리 보상 크기) -> 승자 자동 선정 -> Stage B(승자 고정 후 한 값씩 변경) -> results.md / best_model 생성.
이미 experiments/{exp_name}/eval.json 이 있는 실험은 건너뛰므로, 중단되어도 다시 실행하면 이어서 진행됩니다.

    python -u run_experiments.py            # 본 실험 (episodes=7000)
    python -u run_experiments.py --smoke    # 동작 확인용 (episodes=20, smoke_ 접두사, 결과물은 experiments/ 아래)
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SEED = 0
EVAL_GAMES = 5
EVAL_SEED = 1000
MILESTONE_GAMES = 30   # --extend 모드에서 시점별 체크포인트를 평가하는 판 수

# 기준(기본) 하이퍼파라미터. 실험은 여기서 한 값씩만 바꾼다.
BASE = dict(dist_reward=0.1, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2,
            update_timestep=2000, entropy_coef=0.01)

STAGE_A = [(f"A_dist{v}", {"dist_reward": v}) for v in (0, 0.1, 0.2, 0.3, 0.5, 1.0)]
STAGE_B = [
    ("B_lr0.001", {"lr": 0.001}),
    ("B_lr0.0001", {"lr": 0.0001}),
    ("B_upd4000", {"update_timestep": 4000}),
    ("B_upd1000", {"update_timestep": 1000}),
    ("B_ent0.05", {"entropy_coef": 0.05}),
    ("B_ent0.001", {"entropy_coef": 0.001}),
]

NOTES_LOCK = threading.RLock()
LOG_LOCK = threading.Lock()
STATE = {"failure_header_written": False}


def log(msg):
    with LOG_LOCK:  # 여러 스레드가 동시에 출력해도 줄이 섞이지 않게 함
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def fmt_value(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


class Runner:
    def __init__(self, smoke, train_timeout_hours, rerun_episodes=None):
        self.smoke = smoke
        self.prefix = "smoke_" if smoke else ""
        self.episodes = rerun_episodes or (20 if smoke else 7000)
        self.rerun = rerun_episodes is not None
        self.train_timeout = train_timeout_hours * 3600
        self.concurrency = max(1, min(6, (os.cpu_count() or 2) // 2))

        # 스모크 테스트는 본 결과물(NOTES.md, results.md, best_model/)을 건드리지 않고 experiments/ 아래에 둔다
        if smoke:
            self.notes_path = ROOT / "experiments" / "smoke_NOTES.md"
            self.results_path = ROOT / "experiments" / "smoke_results.md"
            self.best_dir = ROOT / "experiments" / "smoke_best_model"
            self.pid_path = ROOT / "logs" / "smoke_orchestrator.pid"
        else:
            self.notes_path = ROOT / "NOTES.md"
            self.results_path = ROOT / "results.md"
            self.best_dir = ROOT / "best_model"
            self.pid_path = ROOT / "logs" / "orchestrator.pid"
        if self.rerun:  # 재학습 모드는 기존 results.md / best_model 을 건드리지 않고 별도 파일에 저장
            name = f"results_ep{self.episodes}.md"
            self.results_path = ROOT / "experiments" / f"smoke_{name}" if smoke else ROOT / name
            self.pid_path = self.pid_path.with_name(self.pid_path.stem + "_rerun.pid")

        venv_py = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.py = str(venv_py) if venv_py.exists() else sys.executable

        self.env = os.environ.copy()
        self.env.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", OMP_NUM_THREADS="1",
                        PYTHONIOENCODING="utf-8", PYGAME_HIDE_SUPPORT_PROMPT="1")

        self.experiments = {}   # exp_name -> dict(stage, name, changed, params)
        self.failures = []      # (exp_name, phase)

    # ---------- 경로 ----------
    def exp_dir(self, name):
        return ROOT / "experiments" / name

    def model_rel(self, name):
        return f"saved_models/{name}/ppo_snake_final.pth"

    def log_path(self, name):
        return ROOT / "logs" / f"{name}.log"

    # ---------- NOTES.md ----------
    def append_notes(self, text):
        with NOTES_LOCK:
            self.notes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.notes_path, "a", encoding="utf-8") as f:
                f.write(text)

    def record_failure(self, name, phase, returncode):
        tail = ""
        try:
            lines = self.log_path(name).read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-30:])
        except OSError:
            pass
        with NOTES_LOCK:
            self.failures.append((name, phase))
            header = ""
            if not STATE["failure_header_written"]:
                STATE["failure_header_written"] = True
                header = "\n## 실패한 실험 (오케스트레이터 자동 기록)\n"
            self.append_notes(
                f"{header}\n### {name} — {phase} 실패 (returncode={returncode}, {datetime.now():%Y-%m-%d %H:%M:%S})\n"
                f"로그 마지막 30줄 (`logs/{name}.log`):\n\n```\n{tail}\n```\n")
        log(f"FAIL {name} ({phase}, returncode={returncode})")

    # ---------- 서브프로세스 ----------
    def run_logged(self, name, cmd, timeout=None):
        """cmd를 실행하고 stdout/stderr를 logs/{name}.log 에 이어 쓴다. returncode 반환 (타임아웃이면 -999)."""
        lp = self.log_path(name)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== [{datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(cmd)} =====\n")
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=self.env, stdout=lf, stderr=subprocess.STDOUT)
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                lf.write(f"\n===== 타임아웃({timeout}s): SIGINT 로 종료 시도 =====\n")
                lf.flush()
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return -999

    def run_experiment(self, exp_name):
        exp = self.experiments[exp_name]
        ed = self.exp_dir(exp_name)
        eval_path = ed / "eval.json"
        result_path = ed / "result.json"

        if load_json(eval_path) is not None:
            log(f"SKIP {exp_name} (eval.json 이미 존재)")
            return
        try:
            trained = load_json(result_path) is not None and (ROOT / self.model_rel(exp_name)).exists()
            if not trained:
                p = exp["params"]
                cmd = [self.py, "-u", "-m", "main", "--exp_name", exp_name,
                       "--episodes", str(self.episodes), "--seed", str(exp["seed"]),
                       "--dist_reward", str(p["dist_reward"]), "--lr", str(p["lr"]),
                       "--gamma", str(p["gamma"]), "--epochs", str(p["epochs"]),
                       "--eps_clip", str(p["eps_clip"]), "--update_timestep", str(p["update_timestep"]),
                       "--entropy_coef", str(p["entropy_coef"])]
                # 이전 시도(중단/실패)가 남긴 폴더가 있으면 재시작 허용 (이 스크립트가 만든 이름에만 해당)
                if any(x.exists() for x in (ROOT / "runs" / exp_name, ROOT / "saved_models" / exp_name, ed)):
                    cmd.append("--allow_existing")
                    log(f"주의: {exp_name} 이전 시도의 잔여 폴더가 있어 --allow_existing 로 재시작")
                log(f"START {exp_name} ({exp['changed']})")
                t0 = time.time()
                rc = self.run_logged(exp_name, cmd, timeout=self.train_timeout)
                if rc != 0 or load_json(result_path) is None:
                    self.record_failure(exp_name, "train" if rc != -999 else "train(timeout)", rc)
                    return
                log(f"TRAINED {exp_name} ({(time.time() - t0) / 60:.1f}분)")

            cmd = [self.py, "-u", "evaluate.py", "--model", self.model_rel(exp_name),
                   "--games", str(EVAL_GAMES), "--seed", str(EVAL_SEED),
                   "--out", f"experiments/{exp_name}/eval.json"]
            rc = self.run_logged(exp_name, cmd, timeout=1800)
            if rc != 0 or load_json(eval_path) is None:
                self.record_failure(exp_name, "evaluate", rc)
                return
            log(f"DONE {exp_name}")
        except Exception as e:  # 한 실험의 예기치 못한 오류가 전체를 멈추지 않게 한다
            self.append_notes(f"\n### {exp_name} — 오케스트레이터 내부 오류: {e!r}\n")
            self.failures.append((exp_name, "orchestrator"))
            log(f"FAIL {exp_name} (orchestrator error: {e!r})")

    def run_stage(self, names):
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            list(pool.map(self.run_experiment, names))

    # ---------- 실험 등록 / 요약 ----------
    def register(self, stage, base_name, overrides, params, changed, seed=SEED):
        name = self.prefix + base_name
        self.experiments[name] = dict(stage=stage, name=name, overrides=overrides, params=params,
                                      changed=changed, seed=seed)
        return name

    def summary(self, name):
        ev = load_json(self.exp_dir(name) / "eval.json")
        res = load_json(self.exp_dir(name) / "result.json")
        if ev is None or res is None:
            return None
        return dict(name=name, eval_mean=ev["mean"], eval_max=ev["max"], eval_min=ev["min"],
                    eval_scores=ev["scores"], last500=res["last500_mean_score"], minutes=res["total_minutes"])

    def rank(self, names):
        """평가 5판 평균(1순위) -> last500 평균(2순위) 내림차순. 완전 동률이면 등록 순서."""
        rows = [self.summary(n) for n in names]
        rows = [r for r in rows if r is not None]
        return sorted(rows, key=lambda r: (-r["eval_mean"], -r["last500"]))

    # ---------- results.md / best_model ----------
    def table(self, rows):
        head = ("| stage | exp_name | 바꾼 값 | last500 평균 점수 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |\n"
                "|---|---|---|---|---|---|---|---|\n")
        lines = []
        for stage, name, changed in rows:
            s = self.summary(name)
            if s is None:
                lines.append(f"| {stage} | {name} | {changed} | 실패/미완료 | - | - | - | - |")
            else:
                lines.append(f"| {stage} | {name} | {changed} | {s['last500']:.2f} | {s['eval_mean']:.1f} | "
                             f"{s['eval_max']} | {s['eval_min']} | {s['minutes']:.1f} |")
        return head + "\n".join(lines)

    def write_results(self, a_names, b_names, winner_name):
        out = ["# 하이퍼파라미터 실험 결과\n",
               f"- 공통 설정: episodes={self.episodes}, seed={SEED} (실험당 1회), 평가는 최종 모델로 {EVAL_GAMES}판 "
               f"(seed {EVAL_SEED}+판번호)",
               f"- 기본값: " + ", ".join(f"{k}={fmt_value(v)}" for k, v in BASE.items()),
               f"- 생성 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
               "## Stage A: 거리 보상 크기 (대칭 ±x, 나머지 기본값)\n"]
        out.append(self.table([("A", n, self.experiments[n]["changed"]) for n in a_names]))
        ranked_a = self.rank(a_names)
        if ranked_a:
            w = ranked_a[0]
            line = f"\n승자: **{w['name']}** (평가 평균 {w['eval_mean']:.1f}, last500 {w['last500']:.2f})"
            if len(ranked_a) > 1:
                r = ranked_a[1]
                line += (f" / 2위: {r['name']} (평가 평균 {r['eval_mean']:.1f}, last500 {r['last500']:.2f})"
                         f" / 격차: 평가 평균 {w['eval_mean'] - r['eval_mean']:.1f}, "
                         f"last500 {w['last500'] - r['last500']:.2f}")
            out.append(line + "\n")

        if winner_name:
            wd = self.experiments[winner_name]["params"]["dist_reward"]
            out.append(f"\n## Stage B: 승자 dist_reward={fmt_value(wd)} 고정, 한 값씩 변경 (기준선 = {winner_name})\n")
            rows = [("B", winner_name, "(기준선, 변경 없음)")] + \
                   [("B", n, self.experiments[n]["changed"]) for n in b_names]
            out.append(self.table(rows))
        else:
            out.append("\n## Stage B\n\n아직 실행되지 않았습니다 (Stage A 미완료이거나 유효한 결과가 없음).\n")

        candidates = self.rank(a_names + b_names)
        if candidates:
            best = candidates[0]["name"]
            self.best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / self.model_rel(best), self.best_dir / "ppo_snake_best.pth")
            shutil.copy2(self.exp_dir(best) / "config.json", self.best_dir / "config.json")
            cfg = load_json(self.exp_dir(best) / "config.json") or {}
            cfg_text = ", ".join(f"{k}={fmt_value(v)}" for k, v in cfg.items() if k != "policy_path")
            best_rel = self.best_dir.relative_to(ROOT).as_posix()
            out.append(f"\n최고 모델: {best}, 경로: `{best_rel}/ppo_snake_best.pth` "
                       f"(원본 `{self.model_rel(best)}`), 설정: {cfg_text}\n")
        out.append("\n※ 실험당 seed 1개, 평가 5판이라 우연의 영향이 있어 작은 점수 차이는 유의미하지 않을 수 있습니다.\n")
        atomic_write(self.results_path, "\n".join(out))
        log(f"results 저장: {self.results_path.relative_to(ROOT)}")

    # ---------- 전체 흐름 ----------
    def run(self):
        a_names = [self.register("A", n, o, {**BASE, **o}, ", ".join(f"{k}={fmt_value(v)}" for k, v in o.items()))
                   for n, o in STAGE_A]
        b_names = []
        winner_name = None
        started = time.time()
        log(f"오케스트레이터 시작: smoke={self.smoke}, episodes={self.episodes}, 동시 실행 {self.concurrency}, python={self.py}")
        try:
            log("=== Stage A 시작 ===")
            self.run_stage(a_names)
            ranked = self.rank(a_names)
            if ranked:
                w = ranked[0]
                winner_name = w["name"]
                note = (f"\n## Stage A 승자 선정 (오케스트레이터 자동 기록)\n\n"
                        f"- 기준: 평가 5판 평균(1순위), 동점이면 last500 평균(2순위), 그래도 같으면 목록 앞쪽 실험\n"
                        f"- 승자: {w['name']} (평가 평균 {w['eval_mean']:.1f} {w['eval_scores']}, last500 {w['last500']:.2f})\n")
                if len(ranked) > 1:
                    r = ranked[1]
                    note += (f"- 2위: {r['name']} (평가 평균 {r['eval_mean']:.1f} {r['eval_scores']}, last500 {r['last500']:.2f})\n"
                             f"- 격차: 평가 평균 {w['eval_mean'] - r['eval_mean']:.1f}, last500 {w['last500'] - r['last500']:.2f}\n")
                note += "- 전체 순위: " + " > ".join(f"{x['name']}({x['eval_mean']:.1f}/{x['last500']:.2f})" for x in ranked) + "\n"
                self.append_notes(note)
                log(f"Stage A 승자: {winner_name}")

                wd = self.experiments[winner_name]["params"]["dist_reward"]
                for n, o in STAGE_B:
                    b_names.append(self.register("B", n, o, {**BASE, "dist_reward": wd, **o},
                                                 ", ".join(f"{k}={fmt_value(v)}" for k, v in o.items())))
                self.write_results(a_names, b_names, winner_name)  # 중간 저장 (세션/프로세스가 끊겨도 남도록)
                log("=== Stage B 시작 ===")
                self.run_stage(b_names)
            else:
                self.append_notes("\n## Stage A 결과 없음\n\n유효한 Stage A 결과가 없어 Stage B 를 실행하지 않았습니다.\n")
                log("Stage A 유효 결과 없음 -> Stage B 생략")
        finally:
            try:
                self.write_results(a_names, b_names, winner_name)
            except Exception as e:
                log(f"results 작성 실패: {e!r}")
            total = len(a_names) + len(b_names)
            ok = sum(1 for n in a_names + b_names if self.summary(n) is not None)
            self.append_notes(f"\n## 실행 종료 (오케스트레이터 자동 기록)\n\n"
                              f"- 종료 시각: {datetime.now():%Y-%m-%d %H:%M:%S}, 이번 실행 소요: {(time.time() - started) / 60:.1f}분\n"
                              f"- 완료 {ok}/{total}, 이번 실행 중 실패: {self.failures if self.failures else '없음'}\n")
            log(f"오케스트레이터 종료: 완료 {ok}/{total}, 실패 {self.failures}")


    # ---------- 재학습 모드 (--rerun_top) ----------
    def write_rerun_results(self, pairs):
        head = ("| 순위 | 원본 exp (바꾼 값) | 원본 episodes | 원본 last500 | 원본 평가 평균 (최고/최저) | "
                f"재학습 exp | last500 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        lines = []
        for rank, (orig, new) in enumerate(pairs, 1):
            o, n = self.summary(orig), self.summary(new)
            cfg = load_json(self.exp_dir(orig) / "config.json") or {}
            changed = ", ".join(f"{k}={fmt_value(cfg[k])}" for k in BASE if k in cfg and cfg[k] != BASE[k]) or "기본값"
            orig_eps = (load_json(self.exp_dir(orig) / "result.json") or {}).get("episodes", "?")
            left = (f"| {rank} | {orig} ({changed}) | {orig_eps} | {o['last500']:.2f} | "
                    f"{o['eval_mean']:.1f} ({o['eval_max']}/{o['eval_min']}) | ") if o else f"| {rank} | {orig} | ? | ? | ? | "
            right = (f"{new} | {n['last500']:.2f} | {n['eval_mean']:.1f} | {n['eval_max']} | {n['eval_min']} | "
                     f"{n['minutes']:.1f} |") if n else f"{new} | 실패/미완료 | - | - | - | - |"
            lines.append(left + right)
        text = "\n".join([
            f"# 상위 {len(pairs)}개 재학습 결과 (episodes={self.episodes})\n",
            f"- 원본 = Stage A/B 전체에서 평가 5판 평균(1순위)·last500(2순위)으로 뽑은 상위 {len(pairs)}개 (기존 results.md 기준)",
            f"- 설정은 원본 config.json 과 같고 episodes 만 {self.episodes} 로 바꿈. seed={SEED}, 평가는 최종 모델 {EVAL_GAMES}판 (seed {EVAL_SEED}+판번호)",
            f"- 생성 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
            head + "\n".join(lines),
            "\n※ 실험당 seed 1개, 평가 5판이라 우연의 영향이 있어 작은 점수 차이는 유의미하지 않을 수 있습니다.\n"])
        atomic_write(self.results_path, text)
        log(f"results 저장: {self.results_path.relative_to(ROOT)}")

    def run_rerun(self, top_n):
        """기존 Stage A/B 상위 top_n 실험을 같은 설정·seed 로 self.episodes 만큼 다시 학습한다."""
        orig_names = [self.prefix + n for n, _ in STAGE_A + STAGE_B]
        pairs = []  # (원본 이름, 새 이름)
        for row in self.rank(orig_names)[:top_n]:
            orig = row["name"]
            cfg = load_json(self.exp_dir(orig) / "config.json")
            params = {k: cfg[k] for k in BASE}
            base = orig[len(self.prefix):]
            new = self.register("C", f"C_{base}_ep{self.episodes}", {}, params,
                                f"{orig} 설정, episodes={self.episodes}")
            pairs.append((orig, new))
        started = time.time()
        log(f"재학습 시작: 상위 {len(pairs)}개 {[p[0] for p in pairs]} -> episodes={self.episodes}, 동시 실행 {self.concurrency}")
        try:
            self.run_stage([new for _, new in pairs])
        finally:
            try:
                self.write_rerun_results(pairs)
            except Exception as e:
                log(f"results 작성 실패: {e!r}")
            ok = sum(1 for _, new in pairs if self.summary(new) is not None)
            self.append_notes(f"\n## 상위 {len(pairs)}개 재학습, episodes={self.episodes} (오케스트레이터 자동 기록)\n\n"
                              f"- 대상: {', '.join(f'{o} -> {n}' for o, n in pairs)}\n"
                              f"- 결과 표: `{self.results_path.relative_to(ROOT).as_posix()}`, "
                              f"종료 시각 {datetime.now():%Y-%m-%d %H:%M:%S}, 소요 {(time.time() - started) / 60:.1f}분\n"
                              f"- 완료 {ok}/{len(pairs)}, 이번 실행 중 실패: {self.failures if self.failures else '없음'}\n")
            log(f"재학습 종료: 완료 {ok}/{len(pairs)}, 실패 {self.failures}")


    # ---------- 에피소드 확대 모드 (--extend) ----------
    def tb_scores(self, name):
        """TensorBoard 기록에서 {에피소드: 사과 수} 를 읽는다."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(ROOT / "runs" / name), size_guidance={"scalars": 0})
        ea.Reload()
        return {e.step: e.value for e in ea.Scalars("Performance/Score (Apples)")}

    def eval_checkpoint(self, item):
        """saved_models/{name}/ppo_snake_ep{m}_score*.pth 체크포인트를 MILESTONE_GAMES 판 평가해 eval_ep{m}.json 에 저장."""
        name, m = item
        out = self.exp_dir(name) / f"eval_ep{m}.json"
        if load_json(out) is not None:
            return
        ckpts = sorted((ROOT / "saved_models" / name).glob(f"ppo_snake_ep{m}_score*.pth"))
        if not ckpts:
            log(f"체크포인트 없음: {name} ep{m}")
            return
        cmd = [self.py, "-u", "evaluate.py", "--model", ckpts[0].relative_to(ROOT).as_posix(),
               "--games", str(MILESTONE_GAMES), "--seed", str(EVAL_SEED), "--out", f"experiments/{name}/eval_ep{m}.json"]
        rc = self.run_logged(name, cmd, timeout=3600)
        if rc != 0 or load_json(out) is None:
            self.record_failure(name, f"evaluate ep{m}", rc)

    def write_extend_results(self, base, names, milestones):
        seeds = [self.experiments[n]["seed"] for n in names]
        scores = {n: self.tb_scores(n) for n in names}

        def last500(n, m):
            window = [scores[n][s] for s in range(m - 499, m + 1) if s in scores[n]]
            return sum(window) / len(window) if len(window) == 500 else None

        def table(cell, mean_fmt):
            head = "| 에피소드 | " + " | ".join(f"seed {s}" for s in seeds) + " | seed 평균 |\n|---|" + "---|" * (len(seeds) + 1) + "\n"
            rows = []
            for m in milestones:
                vals = [cell(n, m) for n in names]
                nums = [v[0] for v in vals if v is not None]
                mean = mean_fmt(sum(nums) / len(nums)) if nums else "-"
                rows.append(f"| {m} | " + " | ".join(v[1] if v is not None else "-" for v in vals) + f" | {mean} |")
            return head + "\n".join(rows)

        def train_cell(n, m):
            v = last500(n, m)
            return None if v is None else (v, f"{v:.1f}")

        def eval_cell(n, m):
            ev = load_json(self.exp_dir(n) / f"eval_ep{m}.json")
            return None if ev is None else (ev["mean"], f"{ev['mean']:.1f} ({ev['min']}~{ev['max']})")

        orig_cfg = load_json(self.exp_dir(self.prefix + base) / "config.json") or {}
        cfg_text = ", ".join(f"{k}={fmt_value(v)}" for k, v in orig_cfg.items() if k in BASE)
        text = "\n".join([
            f"# {base} 설정으로 에피소드를 늘렸을 때 (episodes={self.episodes}, seed {seeds})\n",
            f"- 설정(원본 {self.prefix + base} 의 config): {cfg_text}",
            f"- 학습 중 저장된 체크포인트(500판마다)를 각 시점에서 {MILESTONE_GAMES}판씩 평가 (seed {EVAL_SEED}+판번호, 같은 조건). "
            f"체크포인트는 같은 실행이 그 시점에 저장한 정책이므로, 에피소드 수만 다르게 따로 돌린 것과 같은 결과임(seed 고정, 결정적).",
            f"- 생성 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
            "## 학습 중 직전 500판 평균 사과 수\n",
            table(train_cell, lambda v: f"{v:.1f}"),
            f"\n## 체크포인트 평가 {MILESTONE_GAMES}판 평균 사과 수 (최저~최고)\n",
            table(eval_cell, lambda v: f"{v:.1f}"),
            "\n※ seed 수가 적고(위 표 참조) 평가 판 수도 유한해 작은 차이는 우연일 수 있습니다.\n"])
        atomic_write(self.results_path, text)
        log(f"results 저장: {self.results_path.relative_to(ROOT)}")

    def run_extend(self, base, seeds, milestones):
        """base 실험(예: A_dist0)의 설정으로 self.episodes 까지, 여러 seed 를 병렬 학습하고 시점별 성능을 평가한다."""
        orig = self.prefix + base
        cfg = load_json(self.exp_dir(orig) / "config.json")
        if cfg is None:
            sys.exit(f"{orig} 의 config.json 이 없습니다.")
        params = {k: cfg[k] for k in BASE}
        milestones = sorted(m for m in milestones if m <= self.episodes)
        names = [self.register("D", f"D_{base}_ep{self.episodes}_s{s}", {}, params,
                               f"{orig} 설정, episodes={self.episodes}, seed={s}", seed=s) for s in seeds]
        name = f"results_extend_{base}_ep{self.episodes}.md"
        self.results_path = ROOT / "experiments" / f"smoke_{name}" if self.smoke else ROOT / name
        started = time.time()
        log(f"확대 실험 시작: {orig} 설정, seed {seeds}, episodes={self.episodes}, 평가 시점 {milestones}")
        try:
            self.run_stage(names)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                list(pool.map(self.eval_checkpoint, [(n, m) for n in names for m in milestones]))
        finally:
            try:
                self.write_extend_results(base, names, milestones)
            except Exception as e:
                log(f"results 작성 실패: {e!r}")
            ok = sum(1 for n in names if load_json(self.exp_dir(n) / "result.json") is not None)
            self.append_notes(f"\n## {base} 설정 에피소드 확대, episodes={self.episodes}, seed {seeds} (오케스트레이터 자동 기록)\n\n"
                              f"- 결과 표: `{self.results_path.relative_to(ROOT).as_posix()}`, "
                              f"종료 시각 {datetime.now():%Y-%m-%d %H:%M:%S}, 소요 {(time.time() - started) / 60:.1f}분\n"
                              f"- 학습 완료 {ok}/{len(names)}, 이번 실행 중 실패: {self.failures if self.failures else '없음'}\n")
            log(f"확대 실험 종료: 학습 완료 {ok}/{len(names)}, 실패 {self.failures}")


def acquire_pidfile(path):
    """중복 실행 방지. pid 파일은 지우지 않고(파일 삭제 금지), 살아 있는 오케스트레이터인지 확인한 뒤 덮어쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = int(path.read_text().strip())
            cmdline = subprocess.run(["ps", "-p", str(old), "-o", "command="],
                                     capture_output=True, text=True).stdout
            if "run_experiments" in cmdline:
                sys.exit(f"이미 실행 중인 오케스트레이터가 있습니다 (pid {old}). 종료합니다.")
        except ValueError:
            pass  # 내용이 깨진 pid 파일이면 무시하고 덮어씀
    path.write_text(str(os.getpid()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="동작 확인용: episodes=20, 실험 이름에 smoke_ 접두사")
    parser.add_argument("--train_timeout_hours", type=float, default=6.0, help="실험 1개 학습 최대 시간")
    parser.add_argument("--rerun_top", type=int, default=0,
                        help="기존 Stage A/B 상위 N개를 같은 설정으로 --episodes 만큼 다시 학습 (예: --rerun_top 3 --episodes 10000)")
    parser.add_argument("--episodes", type=int, default=None, help="--rerun_top 과 함께 사용하는 에피소드 수")
    parser.add_argument("--extend", type=str, default=None,
                        help="이 실험(예: A_dist0)의 설정으로 --episodes 까지 --seeds 각각 학습하고 --milestones 시점의 체크포인트를 평가")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="--extend 에서 사용할 seed 목록")
    parser.add_argument("--milestones", type=int, nargs="+", default=None,
                        help="--extend 에서 평가할 에피소드 시점 (500의 배수). 기본: 5000, 7000, 10000 이후 5000 간격")
    args = parser.parse_args()
    if (args.rerun_top or args.extend) and not args.episodes:
        parser.error("--rerun_top / --extend 는 --episodes 가 필요합니다")

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # 터미널이 닫혀도 계속 실행

    runner = Runner(args.smoke, args.train_timeout_hours,
                    rerun_episodes=args.episodes if (args.rerun_top or args.extend) else None)
    acquire_pidfile(runner.pid_path)
    if args.extend:
        default_ms = sorted({5000, 7000} | set(range(10000, args.episodes + 1, 5000)))
        runner.run_extend(args.extend, args.seeds, args.milestones or default_ms)
    elif args.rerun_top:
        runner.run_rerun(args.rerun_top)
    else:
        runner.run()
