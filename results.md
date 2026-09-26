# 하이퍼파라미터 실험 결과

- 공통 설정: episodes=7000, seed=0 (실험당 1회), 평가는 최종 모델로 5판 (seed 1000+판번호)
- 기본값: dist_reward=0.1, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2, update_timestep=2000, entropy_coef=0.01
- 생성 시각: 2026-09-26 17:59:33

## Stage A: 거리 보상 크기 (대칭 ±x, 나머지 기본값)

| stage | exp_name | 바꾼 값 | last500 평균 점수 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |
|---|---|---|---|---|---|---|---|
| A | A_dist0 | dist_reward=0 | 75.22 | 70.2 | 99 | 36 | 12.3 |
| A | A_dist0.1 | dist_reward=0.1 | 48.13 | 36.6 | 58 | 9 | 10.8 |
| A | A_dist0.2 | dist_reward=0.2 | 31.40 | 32.6 | 59 | 1 | 8.3 |
| A | A_dist0.3 | dist_reward=0.3 | 31.44 | 37.4 | 48 | 27 | 6.7 |
| A | A_dist0.5 | dist_reward=0.5 | 22.16 | 21.0 | 40 | 9 | 5.1 |
| A | A_dist1.0 | dist_reward=1 | 35.38 | 38.2 | 54 | 17 | 7.1 |

승자: **A_dist0** (평가 평균 70.2, last500 75.22) / 2위: A_dist1.0 (평가 평균 38.2, last500 35.38) / 격차: 평가 평균 32.0, last500 39.85


## Stage B: 승자 dist_reward=0 고정, 한 값씩 변경 (기준선 = A_dist0)

| stage | exp_name | 바꾼 값 | last500 평균 점수 | 평가 5판 평균 | 최고 | 최저 | 학습 시간(분) |
|---|---|---|---|---|---|---|---|
| B | A_dist0 | (기준선, 변경 없음) | 75.22 | 70.2 | 99 | 36 | 12.3 |
| B | B_lr0.001 | lr=0.001 | 37.23 | 36.2 | 45 | 32 | 18.0 |
| B | B_lr0.0001 | lr=0.0001 | 0.24 | 0.0 | 0 | 0 | 3.5 |
| B | B_upd4000 | update_timestep=4000 | 17.05 | 17.6 | 27 | 7 | 5.0 |
| B | B_upd1000 | update_timestep=1000 | 56.19 | 51.8 | 71 | 35 | 15.8 |
| B | B_ent0.05 | entropy_coef=0.05 | 11.14 | 7.0 | 9 | 4 | 5.1 |
| B | B_ent0.001 | entropy_coef=0.001 | 0.00 | 0.0 | 0 | 0 | 4.9 |

최고 모델: A_dist0, 경로: `best_model/ppo_snake_best.pth` (원본 `saved_models/A_dist0/ppo_snake_final.pth`), 설정: exp_name=A_dist0, episodes=7000, seed=0, dist_reward=0, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2, update_timestep=2000, entropy_coef=0.01


※ 실험당 seed 1개, 평가 5판이라 우연의 영향이 있어 작은 점수 차이는 유의미하지 않을 수 있습니다.

※ 갱신: 이 표는 기존 상태(basic, 11개) 실험 기준입니다. 이후 몸통 정보 상태(body, 21개) 실험(`results_state_body.md`, `results_state_parity.md`)에서 더 높은 모델이 나와 `best_model/` 은 `E_body_dist0.1_s0` (원본 `saved_models/E_body_dist0.1_s0/ppo_snake_final.pth`, 새 난수 100판 평가 평균 89.9)로 교체했습니다. 위 "최고 모델" 줄의 A_dist0 원본은 `saved_models/A_dist0/` 에 그대로 있습니다.
