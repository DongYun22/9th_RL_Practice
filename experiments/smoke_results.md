# 하이퍼파라미터 실험 결과

- 공통 설정: episodes=20, seed=0 (실험당 1회), 평가는 최종 모델로 5판 (seed 1000+판번호)
- 기본값: dist_reward=0.1, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2, update_timestep=2000, entropy_coef=0.01
- 생성 시각: 2026-09-26 17:27:49

## Stage A: 거리 보상 크기 (대칭 ±x, 나머지 기본값)

| stage | exp_name | 바꾼 값 | last500 평균 점수 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |
|---|---|---|---|---|---|---|---|
| A | smoke_A_dist0 | dist_reward=0 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| A | smoke_A_dist0.1 | dist_reward=0.1 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| A | smoke_A_dist0.2 | dist_reward=0.2 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| A | smoke_A_dist0.3 | dist_reward=0.3 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| A | smoke_A_dist0.5 | dist_reward=0.5 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| A | smoke_A_dist1.0 | dist_reward=1 | 0.10 | 0.0 | 0 | 0 | 0.0 |

승자: **smoke_A_dist0** (평가 평균 0.0, last500 0.10) / 2위: smoke_A_dist0.1 (평가 평균 0.0, last500 0.10) / 격차: 평가 평균 0.0, last500 0.00


## Stage B: 승자 dist_reward=0 고정, 한 값씩 변경 (기준선 = smoke_A_dist0)

| stage | exp_name | 바꾼 값 | last500 평균 점수 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |
|---|---|---|---|---|---|---|---|
| B | smoke_A_dist0 | (기준선, 변경 없음) | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_lr0.001 | lr=0.001 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_lr0.0001 | lr=0.0001 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_upd4000 | update_timestep=4000 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_upd1000 | update_timestep=1000 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_ent0.05 | entropy_coef=0.05 | 0.10 | 0.0 | 0 | 0 | 0.0 |
| B | smoke_B_ent0.001 | entropy_coef=0.001 | 0.10 | 0.0 | 0 | 0 | 0.0 |

최고 모델: smoke_A_dist0, 경로: `experiments/smoke_best_model/ppo_snake_best.pth` (원본 `saved_models/smoke_A_dist0/ppo_snake_final.pth`), 설정: exp_name=smoke_A_dist0, episodes=20, seed=0, dist_reward=0, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2, update_timestep=2000, entropy_coef=0.01


※ 실험당 seed 1개, 평가 5판이라 우연의 영향이 있어 작은 점수 차이는 유의미하지 않을 수 있습니다.
