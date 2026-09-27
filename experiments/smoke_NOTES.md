
## Stage A 승자 선정 (오케스트레이터 자동 기록)

- 기준: 평가 5판 평균(1순위), 동점이면 last500 평균(2순위), 그래도 같으면 목록 앞쪽 실험
- 승자: smoke_A_dist0 (평가 평균 0.0 [0, 0, 0, 0, 0], last500 0.10)
- 2위: smoke_A_dist0.1 (평가 평균 0.0 [0, 0, 0, 0, 0], last500 0.10)
- 격차: 평가 평균 0.0, last500 0.00
- 전체 순위: smoke_A_dist0(0.0/0.10) > smoke_A_dist0.1(0.0/0.10) > smoke_A_dist0.2(0.0/0.10) > smoke_A_dist0.3(0.0/0.10) > smoke_A_dist0.5(0.0/0.10) > smoke_A_dist1.0(0.0/0.10)

## 실행 종료 (오케스트레이터 자동 기록)

- 종료 시각: 2026-09-26 17:27:02, 이번 실행 소요: 0.1분
- 완료 12/12, 이번 실행 중 실패: 없음

## 실패한 실험 (오케스트레이터 자동 기록)

### smoke_fail_test — train 실패 (returncode=1, 2026-09-26 17:27:48)
로그 마지막 30줄 (`logs/smoke_fail_test.log`):

```

===== [2026-09-26 17:27:47] /Users/dongyunkwak/GitHub/9th_RL_Practice/.venv/bin/python -u -m main --exp_name smoke_fail_test --episodes 20 --seed 0 --dist_reward 0.1 --lr 0.0003 --gamma 0.99 --epochs 4 --eps_clip 0.2 --update_timestep 0 --entropy_coef 0.01 =====
🚀 PPO 에이전트 학습을 시작합니다...
TensorBoard 실행: tensorboard --logdir=runs
종료하려면 Ctrl+C를 누르세요. (현재까지의 모델이 자동 저장됩니다)
설정: {"exp_name": "smoke_fail_test", "episodes": 20, "seed": 0, "policy_path": "", "dist_reward": 0.1, "lr": 0.0003, "gamma": 0.99, "epochs": 4, "eps_clip": 0.2, "update_timestep": 0, "entropy_coef": 0.01}
⚠️ 불러올 모델이 없습니다. 처음부터 새로 학습을 시작합니다.
현재까지 학습된 모델이 'saved_models/smoke_fail_test/ppo_snake_final.pth'에 저장되었습니다.
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/dongyunkwak/GitHub/9th_RL_Practice/main.py", line 181, in <module>
    train(policy_path=args.policy_path, episodes=args.episodes, exp_name=args.exp_name,
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          dist_reward=args.dist_reward, lr=args.lr, gamma=args.gamma, epochs=args.epochs,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          eps_clip=args.eps_clip, update_timestep=args.update_timestep,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          entropy_coef=args.entropy_coef, seed=args.seed, allow_existing=args.allow_existing)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dongyunkwak/GitHub/9th_RL_Practice/main.py", line 116, in train
    if time_step % update_timestep == 0:
       ~~~~~~~~~~^~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
```

### smoke_fail_test — train 실패 (returncode=1, 2026-09-26 17:27:49)
로그 마지막 30줄 (`logs/smoke_fail_test.log`):

```
          entropy_coef=args.entropy_coef, seed=args.seed, allow_existing=args.allow_existing)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dongyunkwak/GitHub/9th_RL_Practice/main.py", line 116, in train
    if time_step % update_timestep == 0:
       ~~~~~~~~~~^~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero

===== [2026-09-26 17:27:48] /Users/dongyunkwak/GitHub/9th_RL_Practice/.venv/bin/python -u -m main --exp_name smoke_fail_test --episodes 20 --seed 0 --dist_reward 0.1 --lr 0.0003 --gamma 0.99 --epochs 4 --eps_clip 0.2 --update_timestep 0 --entropy_coef 0.01 --allow_existing =====
🚀 PPO 에이전트 학습을 시작합니다...
TensorBoard 실행: tensorboard --logdir=runs
종료하려면 Ctrl+C를 누르세요. (현재까지의 모델이 자동 저장됩니다)
설정: {"exp_name": "smoke_fail_test", "episodes": 20, "seed": 0, "policy_path": "", "dist_reward": 0.1, "lr": 0.0003, "gamma": 0.99, "epochs": 4, "eps_clip": 0.2, "update_timestep": 0, "entropy_coef": 0.01}
⚠️ 불러올 모델이 없습니다. 처음부터 새로 학습을 시작합니다.
현재까지 학습된 모델이 'saved_models/smoke_fail_test/ppo_snake_final.pth'에 저장되었습니다.
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/dongyunkwak/GitHub/9th_RL_Practice/main.py", line 181, in <module>
    train(policy_path=args.policy_path, episodes=args.episodes, exp_name=args.exp_name,
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          dist_reward=args.dist_reward, lr=args.lr, gamma=args.gamma, epochs=args.epochs,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          eps_clip=args.eps_clip, update_timestep=args.update_timestep,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          entropy_coef=args.entropy_coef, seed=args.seed, allow_existing=args.allow_existing)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dongyunkwak/GitHub/9th_RL_Practice/main.py", line 116, in train
    if time_step % update_timestep == 0:
       ~~~~~~~~~~^~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
```

## Stage A 승자 선정 (오케스트레이터 자동 기록)

- 기준: 평가 5판 평균(1순위), 동점이면 last500 평균(2순위), 그래도 같으면 목록 앞쪽 실험
- 승자: smoke_A_dist0 (평가 평균 0.0 [0, 0, 0, 0, 0], last500 0.10)
- 2위: smoke_A_dist0.1 (평가 평균 0.0 [0, 0, 0, 0, 0], last500 0.10)
- 격차: 평가 평균 0.0, last500 0.00
- 전체 순위: smoke_A_dist0(0.0/0.10) > smoke_A_dist0.1(0.0/0.10) > smoke_A_dist0.2(0.0/0.10) > smoke_A_dist0.3(0.0/0.10) > smoke_A_dist0.5(0.0/0.10) > smoke_A_dist1.0(0.0/0.10)

## 실행 종료 (오케스트레이터 자동 기록)

- 종료 시각: 2026-09-26 17:27:49, 이번 실행 소요: 0.0분
- 완료 12/12, 이번 실행 중 실패: 없음

## 상위 3개 재학습, episodes=30 (오케스트레이터 자동 기록)

- 대상: smoke_A_dist0 -> smoke_C_A_dist0_ep30, smoke_A_dist0.1 -> smoke_C_A_dist0.1_ep30, smoke_A_dist0.2 -> smoke_C_A_dist0.2_ep30
- 결과 표: `experiments/smoke_results_ep30.md`, 종료 시각 2026-09-26 18:30:03, 소요 0.1분
- 완료 3/3, 이번 실행 중 실패: 없음

## A_dist0 설정 에피소드 확대, episodes=1000, seed [0, 1] (오케스트레이터 자동 기록)

- 결과 표: `experiments/smoke_results_extend_A_dist0_ep1000.md`, 종료 시각 2026-09-26 20:32:05, 소요 0.5분
- 학습 완료 2/2, 이번 실행 중 실패: 없음

## spec 실험 specsmoke (오케스트레이터 자동 기록)

- 결과 표: `experiments/smoke_results_specsmoke.md`, 종료 시각 2026-09-26 22:33:07, 소요 0.6분
- 학습 완료 2/2, 이번 실행 중 실패: 없음
