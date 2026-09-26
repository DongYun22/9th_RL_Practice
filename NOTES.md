# 실험 노트 (NOTES)

작성 시점: 2026-09-26. 아래 "자동 기록" 표시가 있는 섹션은 `run_experiments.py` 가 실행 중/종료 시 이어서 추가합니다.

## 1. 환경 확인 결과 (지시문 가정과 달랐던 점)

- 실제 환경은 **macOS(Darwin) + zsh** 이고 Windows/PowerShell 이 아니었음. 인터프리터는 `.venv/bin/python` (Python 3.13.15, torch 2.13.0, pygame 2.6.1).
  오케스트레이터는 `.venv/bin/python` (없으면 `.venv/Scripts/python.exe`, 그것도 없으면 현재 인터프리터)을 자동 선택함.
- CPU 12코어 → 동시 실행 수 = max(1, min(6, 12//2)) = **6**. 학습은 CPU 단일 스레드(`OMP_NUM_THREADS=1`, torch 스레드 1개 확인).
- 학습 시작 시점에 돌고 있던 학습 프로세스는 없었음 (TensorBoard 서버만 실행 중이라 새 `runs/` 폴더가 자동으로 대시보드에 나타남).
- `requirements.txt` 는 UTF-16 인코딩이라 그대로는 pip 이 읽기 어려울 수 있음. 새 패키지는 설치하지 않았고 기존 `.venv` 만 사용함.

## 2. 내가 내린 판단

1. **작업 트리의 미커밋 수정 처리**: 시작 시점에 `snake_code.py` 는 거리 보상이 `+0 / -0` 으로, `main.py` 는 episodes=7000 / `runs/snake_ppo_experiment_4` 로 수정(미커밋)되어 있었음.
   지시대로 `dist_reward` 를 인자로 만들고 **기본값을 0.1(커밋된 원래 값)** 로 두었음. 따라서 `python -m main` 기본 동작은 원래 하이퍼파라미터(±0.1)와 같고, 미커밋이던 "거리 보상 0" 상태는 `--dist_reward 0` 으로 재현할 수 있음(Stage A 의 `A_dist0`).
   main.py 의 미커밋 변경(7000, experiment_4 경로)은 새 main.py 가 대체함.
2. **`--episodes` 기본값**: 지시문이 "`--episodes(7000)`" 과 "episodes만 원래 10000 유지" 로 서로 달라, **기본값은 10000(커밋된 원래 값)** 으로 두었음. 모든 실험은 오케스트레이터가 `--episodes 7000` 을 명시해서 넘기므로 실험 결과에는 영향 없음. 기본값을 7000 으로 바꾸려면 `main.py` 의 `--episodes` 기본값과 `train()` 기본 인자 두 곳을 고치면 됨.
3. **`--seed` 기본값 0**: 원래 코드는 seed 가 없었으나 지시대로 기본 0 으로 고정함(같은 설정을 두 번 돌리면 `total_steps` 까지 동일함을 스모크에서 확인).
4. **기존 결과 보호 장치 추가**: `main.py` 는 `runs/{exp_name}`, `saved_models/{exp_name}`, `experiments/{exp_name}` 중 하나라도 이미 있으면 아무것도 쓰지 않고 종료함(예: `--exp_name main` 은 기존 `saved_models/main/` 때문에 거부됨, 확인함).
   중단된 실험을 오케스트레이터가 재시작할 때만 `--allow_existing` 을 붙임(이 스크립트가 만든 A_/B_/smoke_ 이름에만 해당).
5. **서브프로세스 환경변수**: 지시된 `SDL_VIDEODRIVER=dummy`, `OMP_NUM_THREADS=1`, `python -u` 외에 `SDL_AUDIODRIVER=dummy`(오디오 장치 점유 방지), `PYTHONIOENCODING=utf-8`, `PYGAME_HIDE_SUPPORT_PROMPT=1` 을 서브프로세스에만 추가로 설정함(시스템 설정 변경 아님).
6. **평가 방식**: `evaluate.py` 는 play.py 와 같은 방식(PPO 생성 → `load_state_dict` → `eval()` → `policy.act`, 렌더링/sleep 없음)이며, **판마다 `seed+판번호`로 난수를 고정**해서(기본 1000..1004) 모델 간 비교가 같은 조건이 되게 함. 평가 대상은 체크포인트가 아니라 **최종 모델 `ppo_snake_final.pth`**.
7. **Stage A 승자 선정**: 평가 5판 평균 → 동점이면 last500 평균 → 그래도 같으면 목록 앞쪽(작은 dist_reward) 실험. 실패한 실험은 후보에서 제외. 결과는 아래 "Stage A 승자 선정(자동 기록)" 에 기록됨.
8. **Stage B 기준선**: 승자 실험 자체를 다시 돌리지 않고 승자의 결과를 그대로 기준선 행으로 results.md 에 표시함. Stage B 는 승자의 `dist_reward` 를 고정하고 나머지는 기본값에서 한 값씩만 바꿈.
9. **최고 모델**: Stage A+B 전체 중 같은 기준(평가 평균 → last500)으로 1위를 골라 `best_model/ppo_snake_best.pth` 와 `best_model/config.json` 으로 **복사만** 함(원본 유지). 참고: `.gitignore` 가 `ppo_snake_*.pth` 를 무시하므로 커밋하려면 `git add -f` 가 필요함.
10. **학습 타임아웃**: 실험 1개당 학습 최대 6시간(`--train_timeout_hours`). 넘으면 SIGINT 로 종료해서 실패로 기록하고 나머지는 계속 진행함.
11. **파일 삭제 금지 준수**: 오케스트레이터의 pid 파일(`logs/orchestrator.pid`)도 지우지 않고 다음 실행 때 덮어씀(살아있는 프로세스인지 `ps` 로 확인해 중복 실행만 막음). results.md 는 임시파일에 쓴 뒤 rename 으로 교체함.
12. **절전 방지**: 노트북이 유휴 상태로 잠들면 학습이 멈추므로 오케스트레이터를 `caffeinate -ims` 로 감싸 실행함(프로세스 단위 절전 방지이며 시스템 설정 변경은 아님). 전원 어댑터 연결 상태(충전 중)였음. 다만 덮개를 닫으면 잠들 수 있음.
13. **git**: `status`, `diff`, `log` 같은 읽기 전용 명령만 사용함. 커밋/푸시/PR/브랜치 생성은 하지 않았고 git 설정도 건드리지 않음.
14. **README 는 수정하지 않음**: README 의 "main.py 아래쪽에서 policy_path 수정" 설명은 이제 `--policy_path` 인자로 대체 가능하지만(기본 동작은 그대로), 문서 변경은 요청 범위 밖이라 두었음.
15. **TensorBoard 표시**: 새 실험은 `runs/{exp_name}` 으로 저장되므로 기존 `snake_ppo_experiment_*` 와 나란히 보임. 스모크 테스트 폴더(`runs/smoke_*`)도 함께 보이니 필요하면 `--logdir_spec` 등으로 필터링하면 됨.

## 3. 수정/추가한 파일 목록

수정(기존 파일):
- `snake_code.py`: `SnakeGame(dist_reward=0.1)` 생성자 인자 추가, `step()` 의 거리 보상을 `+dist_reward / -dist_reward` 로 변경 (다른 보상은 그대로).
- `PPO_code.py`: `PPO(..., entropy_coef=0.01)` 인자 추가, loss 의 하드코딩된 0.01 대신 사용.
- `main.py`: argparse(`--exp_name --episodes --dist_reward --lr --gamma --epochs --eps_clip --update_timestep --entropy_coef --seed --policy_path` + `--allow_existing`), `runs/{exp_name}`, `saved_models/{exp_name}/`, seed 고정, `writer.add_text('config')`, `experiments/{exp_name}/config.json` · `result.json`, finally 블록 저장 경로 변경. 학습 루프 자체는 그대로.

신규:
- `evaluate.py`: 렌더링 없는 N판 평가 (`--model --games(5) --seed(1000) --out`).
- `run_experiments.py`: Stage A → 승자 선정 → Stage B 오케스트레이터 (`--smoke` 로 동작 확인 모드).
- `NOTES.md`(이 파일), 실행 결과물 `results.md`, `best_model/`, `experiments/`, `logs/`, `runs/{A_*,B_*}`, `saved_models/{A_*,B_*}/` (자동 생성).

수정하지 않은 파일: `play.py`, `README.md`, `requirements.txt`, `.gitignore`. 기존 `runs/snake_ppo_experiment_1~4`, `saved_models/` 의 기존 파일·폴더(`main/`, `*_backup.pth`, `ppo_snake_final*.pth`)는 손대지 않음(스모크 후 파일 수 등 재확인함).

## 4. 스모크 테스트 산출물 (삭제하지 않고 남겨 둠)

확인한 것: main.py(20 에피소드)와 evaluate.py 정상 동작, 같은 seed 두 번 실행 시 `total_steps` 동일(재현성), `dist_reward`/`entropy_coef` 가 실제 보상·loss 에 반영됨(0/0.1/0.5/1.0 각각의 step 보상, 계수별 가중치 차이 확인), 기존 폴더 덮어쓰기 거부, 오케스트레이터 전체 흐름(`--smoke`: 12개 실험, 승자 선정, results 생성, best_model 복사), 재실행 시 12개 모두 SKIP, 학습 실패(`update_timestep=0` 로 의도적 ZeroDivisionError) 시 로그 꼬리를 노트에 기록하고 계속 진행.

- `experiments/smoke_train_a`, `experiments/smoke_train_b`: main.py/evaluate.py 단독 확인
- `experiments/smoke_A_dist{0,0.1,0.2,0.3,0.5,1.0}`, `experiments/smoke_B_{lr0.001,lr0.0001,upd4000,upd1000,ent0.05,ent0.001}`: 오케스트레이터 `--smoke` 12개 실험
- `experiments/smoke_fail_test`: 의도적 실패 실험 (실패 처리 확인용, 그 기록은 `experiments/smoke_NOTES.md` 에 있음)
- `experiments/smoke_results.md`, `experiments/smoke_NOTES.md`, `experiments/smoke_best_model/`
- 위 각 이름에 대응하는 `runs/smoke_*`, `saved_models/smoke_*`, `logs/smoke_*.log`, `logs/smoke_orchestrator.pid`

## 5. 실행/재개/중단 방법

- 본 실행: `caffeinate -ims .venv/bin/python -u run_experiments.py > logs/orchestrator.log 2>&1` (detached 로 실행함). 진행 상황은 `logs/orchestrator.log`, 실험별 로그는 `logs/{exp_name}.log`.
- 끊기거나 중단됐다면 같은 명령을 다시 실행하면 `eval.json` 이 있는 실험은 건너뛰고 이어서 진행함(학습만 끝나고 평가가 안 된 실험은 평가만 다시 함).
- 중단: `pkill -f run_experiments.py; pkill -f "main --exp_name"` (남은 학습 프로세스까지 종료).

## Stage A 승자 선정 (오케스트레이터 자동 기록)

- 기준: 평가 5판 평균(1순위), 동점이면 last500 평균(2순위), 그래도 같으면 목록 앞쪽 실험
- 승자: A_dist0 (평가 평균 70.2 [76, 36, 84, 99, 56], last500 75.22)
- 2위: A_dist1.0 (평가 평균 38.2 [17, 54, 42, 35, 43], last500 35.38)
- 격차: 평가 평균 32.0, last500 39.85
- 전체 순위: A_dist0(70.2/75.22) > A_dist1.0(38.2/35.38) > A_dist0.3(37.4/31.44) > A_dist0.1(36.6/48.13) > A_dist0.2(32.6/31.40) > A_dist0.5(21.0/22.16)

## 실행 종료 (오케스트레이터 자동 기록)

- 종료 시각: 2026-09-26 17:59:33, 이번 실행 소요: 30.4분
- 완료 12/12, 이번 실행 중 실패: 없음

## 6. 최종 요약 및 해석 시 주의 (실행 종료 후 추가)

- **실패한 실험: 없음.** 본 실험 12개 모두 학습·평가 완료, 재시도/재시작 없음. 소요 시간은 17:29~17:59 (약 30분, 동시 6개).
- 최고 모델은 **A_dist0** (dist_reward=0, 나머지 기본값): 평가 5판 평균 70.2 (최고 99 / 최저 36), last500 평균 75.22. 원래 기본 설정(A_dist0.1)은 평가 36.6 / last500 48.13.
- Stage A 는 거리 보상이 없는 쪽이 가장 좋았고 0.1~1.0 사이에서는 뚜렷한 경향이 없었음(평가 21.0~38.2). 이 상태는 시작 시점에 미커밋으로 작업 트리에 있던 "거리 보상 0" 설정과 같음.
- Stage B 는 기준선(A_dist0)보다 나은 변경이 없었음. update_timestep=1000 이 51.8 로 가장 근접했고, `B_lr0.0001`, `B_ent0.001` 은 학습 종료 시점에도 사과를 거의 못 먹음(last500 0.24 / 0.00, 마지막 에피소드가 Reward -14.00 · Steps 401 = 사과 없이 굶어 종료). 거리 보상이 0 이라 보상이 드문 상태에서 학습률/탐험을 줄이면 첫 사과를 찾지 못하는 것으로 **추정**되나 별도로 검증하지는 않았음.
- 실험당 seed 1개, 평가 5판이라 점수 차이가 작은 항목(예: A_dist0.1~1.0 사이)은 우연일 수 있음. 판별 점수 편차도 큼(A_dist0 평가 36~99). 결론을 확정하려면 seed 를 늘려 반복 실행 필요.
- 실행 중 기존 결과물(`runs/snake_ppo_experiment_1~4`, `saved_models/` 의 기존 파일·폴더)은 변경되지 않음(마지막 수정 시각이 이 작업 시작 이전임을 확인).

## 상위 3개 재학습, episodes=10000 (오케스트레이터 자동 기록)

- 대상: A_dist0 -> C_A_dist0_ep10000, B_upd1000 -> C_B_upd1000_ep10000, A_dist1.0 -> C_A_dist1.0_ep10000
- 결과 표: `results_ep10000.md`, 종료 시각 2026-09-26 19:00:27, 소요 30.2분
- 완료 3/3, 이번 실행 중 실패: 없음

## 7. 상위 3개 10000 에피소드 재학습 (추가 요청)

- **대상 선정**: Stage A+B 12개 전체를 기존 기준(평가 5판 평균 → last500)으로 순위 매긴 상위 3개 = `A_dist0`(70.2), `B_upd1000`(51.8), `A_dist1.0`(38.2). 설정은 각자의 `config.json` 그대로, episodes 만 10000, seed=0 유지.
- **실험 이름**: `C_{원본이름}_ep10000` (`C_A_dist0_ep10000`, `C_B_upd1000_ep10000`, `C_A_dist1.0_ep10000`). 기존 폴더와 이름이 겹치지 않아 기존 결과물은 변경되지 않음(재학습 시작 이후 기존 파일 수정 없음 확인).
- **코드 변경**: `run_experiments.py` 에 `--rerun_top N --episodes E` 모드 추가(기본 동작은 그대로). 결과 표는 별도 파일 `results_ep10000.md`, 로그는 `logs/orchestrator_rerun.log`, `logs/C_*.log`. **`results.md` 와 `best_model/` 은 수정하지 않았음.**
- **검증**: seed 가 같아서 재학습의 앞 7000 에피소드 점수는 기존 실행과 3개 모두 7000/7000 동일(TensorBoard 기록 비교). 즉 이번 결과는 기존 실행에 3000 에피소드를 이어 학습한 것과 같음. 옵티마이저 상태는 저장하지 않으므로 이어 학습이 아니라 처음부터 다시 돌렸음.
- **결과** (last500 / 평가 5판 평균, 7000 → 10000): `A_dist0` 75.22 → 80.96 / 70.2 → 65.2, `B_upd1000` 56.19 → 64.44 / 51.8 → 74.0, `A_dist1.0` 35.38 → 38.95 / 38.2 → 36.6. 실패 없음, 소요 약 30분(18:30~19:00).
- **해석 주의**: 1·2위는 두 기준이 엇갈림. 평가 5판 평균은 `B_upd1000`(74.0)이, last500 은 `A_dist0`(80.96)이 앞서며, 평가 5판은 편차가 커서(최저 36 ~ 최고 92) 두 실험은 구분되지 않는 수준으로 보임. 미리 정한 규칙(평가 평균 우선)을 다시 적용하면 최고 모델은 `C_B_upd1000_ep10000` 이 되지만, `best_model/` 은 요청이 없어 갱신하지 않았음.
- **스모크 추가분**(삭제하지 않고 남김): `experiments/smoke_C_A_dist0_ep30`, `smoke_C_A_dist0.1_ep30`, `smoke_C_A_dist0.2_ep30`, `experiments/smoke_results_ep30.md` 와 대응하는 `runs/smoke_C_*`, `saved_models/smoke_C_*`, `logs/smoke_C_*.log`, `logs/smoke_orchestrator_rerun.pid`. `experiments/smoke_NOTES.md` 에도 재학습 기록이 추가됨.

## A_dist0 설정 에피소드 확대, episodes=20000, seed [0, 1, 2] (오케스트레이터 자동 기록)

- 결과 표: `results_extend_A_dist0_ep20000.md`, 종료 시각 2026-09-26 21:50:02, 소요 77.8분
- 학습 완료 3/3, 이번 실행 중 실패: 없음

## 8. A_dist0 설정으로 에피소드 확대 (20000, seed 3개)

- **질문**: A_dist0(거리 보상 0, 나머지 기본값)에서 에피소드를 더 늘리면 성능이 좋아지는가?
- **설계 판단**: 이전 결과가 seed 1개였고 A_dist0 는 보상이 드문 설정이라 seed 에 따라 학습 결과가 갈릴 수 있어서, **seed 0·1·2 를 병렬로 20000 에피소드까지** 학습함(벽시계 시간은 동일). 학습 중 500판마다 저장되는 체크포인트를 **5000/7000/10000/15000/20000 시점에서 30판씩** 평가함(평가 5판은 편차가 커서 변화를 판단하기 어려움). 평가 시드는 기존과 같은 1000+판번호.
- **코드 변경**: `run_experiments.py` 에 `--extend EXP --episodes E --seeds ... [--milestones ...]` 모드 추가(`register()` 가 실험별 seed 를 받도록 변경, 기존 모드의 동작은 그대로). 실험 이름 `D_A_dist0_ep20000_s{seed}`, 결과 표 `results_extend_A_dist0_ep20000.md`, 시점별 평가는 `experiments/{exp}/eval_ep{m}.json`, 로그 `logs/orchestrator_extend.log`. `results.md`, `results_ep10000.md`, `best_model/` 은 수정하지 않음.
- **결과**: 세 seed 모두 학습이 진행되어 실패 없음(소요 약 78분, 가장 느린 seed 0 기준). 학습 중 점수의 구간 평균(seed 평균)은 7001~10000: 54.3 → 10001~15000: 54.0 → 15001~20000: 54.2 로 **7000~10000 에피소드 이후 사실상 정체**. seed 별로는 seed 0 79.1/77.0/78.1, seed 1 55.9/55.9/55.5, seed 2 27.8/29.1/29.1. 30판 평가 평균(seed 평균)은 7000: 55.0, 10000: 53.6, 15000: 57.3, 20000: 57.4 로 +2.4 정도지만 seed 1 의 평가 상승(58→66)은 같은 seed 의 학습 곡선이 평평해서 평가 잡음으로 봄(30판 표준오차 약 4~5, 이 해석은 추가 검증하지 않음).
- **시사점**: 에피소드 수보다 **seed(초기 조건)에 따른 차이가 훨씬 큼**(정체 수준이 seed 별로 약 28 / 56 / 78). 따라서 앞선 7000·10000 에피소드의 순위(A_dist0 vs B_upd1000 등)도 seed 1개로는 확정하기 어렵다는 점이 다시 확인됨.
- **스모크 추가분**(삭제하지 않고 남김): `experiments/smoke_D_A_dist0_ep1000_s0`, `smoke_D_A_dist0_ep1000_s1`, `experiments/smoke_results_extend_A_dist0_ep1000.md` 와 대응하는 `runs/smoke_D_*`, `saved_models/smoke_D_*`, `logs/smoke_D_*.log`.
