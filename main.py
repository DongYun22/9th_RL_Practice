import argparse
import json
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from PPO_code import PPO, RolloutBuffer
from snake_code import SnakeGame

def set_seed(seed):
    # 재현성을 위해 파이썬/넘파이/토치 난수를 모두 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(policy_path = "", episodes = 10000, exp_name = None, dist_reward = 0.1,
          lr = 0.0003, gamma = 0.99, epochs = 4, eps_clip = 0.2,
          update_timestep = 2000, entropy_coef = 0.01, seed = 0, allow_existing = False):
    if exp_name is None:
        exp_name = datetime.now().strftime("manual_%Y%m%d_%H%M%S")

    # 기존 결과물을 덮어쓰지 않도록, 이미 있는 실험 이름이면 시작하지 않음
    log_dir = f"runs/{exp_name}"
    model_dir = f"saved_models/{exp_name}"
    exp_dir = f"experiments/{exp_name}"
    if not allow_existing:
        for path in (log_dir, model_dir, exp_dir):
            if os.path.exists(path):
                sys.exit(f"❌ '{path}' 가 이미 존재합니다. 기존 결과를 보호하기 위해 중단합니다. 다른 --exp_name을 사용하세요.")

    set_seed(seed)

    config = {
        "exp_name": exp_name, "episodes": episodes, "seed": seed, "policy_path": policy_path,
        "dist_reward": dist_reward, "lr": lr, "gamma": gamma, "epochs": epochs,
        "eps_clip": eps_clip, "update_timestep": update_timestep, "entropy_coef": entropy_coef,
    }
    config_text = json.dumps(config, indent=2, ensure_ascii=False)
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/config.json", "w", encoding="utf-8") as f:
        f.write(config_text)

    # TensorBoard 로그를 저장할 디렉토리 지정
    writer = SummaryWriter(log_dir)
    writer.add_text('config', "\n".join("    " + line for line in config_text.splitlines()))

    # 초기화
    env = SnakeGame(dist_reward=dist_reward)
    state_dim = 11  # 예: 뱀의 상태 데이터 크기
    action_dim = 3  # 예: 직진, 좌, 우
    ppo_agent = PPO(state_dim, action_dim, lr=lr, gamma=gamma, epochs=epochs, eps_clip=eps_clip, entropy_coef=entropy_coef)
    memory = RolloutBuffer()

    max_episodes = episodes # 최대 돌리는 에피소드 수
    time_step = 0
    save_interval = 500 # 모델 저장 주기
    scores = []         # 에피소드별 사과 수 (result.json 계산용)
    start_time = time.time()

    print("🚀 PPO 에이전트 학습을 시작합니다...")
    print("TensorBoard 실행: tensorboard --logdir=runs")
    print("종료하려면 Ctrl+C를 누르세요. (현재까지의 모델이 자동 저장됩니다)")
    print(f"설정: {json.dumps(config, ensure_ascii=False)}")

    load_path = policy_path

    if os.path.exists(load_path):
        # 현재 정책에 가중치 로드
        ppo_agent.policy.load_state_dict(torch.load(load_path))

        # 과거 정책에도 동일하게 덮어씌우기
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())

        # 모델을 학습 모드로 설정
        ppo_agent.policy.train()

        print(f"✅ 기존 모델({load_path})을 성공적으로 불러왔습니다. 이어서 학습합니다.")
    else:
        print("⚠️ 불러올 모델이 없습니다. 처음부터 새로 학습을 시작합니다.")

    try:
        for episode in range(1, max_episodes + 1): # 1만 판 반복
            state = env.reset()
            episode_reward = 0  # 한 에피소드의 총 보상
            step_count = 0      # 생존 시간(스텝)

            while True:
                time_step += 1
                step_count += 1

                # 1. 상태를 PyTorch 텐서로 변환하여 에이전트에 전달
                state_tensor = torch.FloatTensor(state)
                action, logprob, _ = ppo_agent.policy.act(state_tensor)

                # 2. 선택한 행동으로 환경(게임) 1스텝 진행
                next_state, reward, done = env.step(action)

                # 3. 버퍼에 현재 스텝의 궤적(Trajectory) 저장
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.dones.append(done)

                episode_reward += reward
                state = next_state

                if time_step % update_timestep == 0:
                    actor_loss, critic_loss = ppo_agent.update(memory)
                    writer.add_scalar('Loss/Actor', actor_loss, time_step)
                    writer.add_scalar('Loss/Critic', critic_loss, time_step)

                # 게임 오버 시 루프 탈출
                if done:
                    break

            scores.append(env.score)
            writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Performance/Score (Apples)', env.score, episode)
            writer.add_scalar('Performance/Survival_Steps', step_count, episode)

            if episode % 50 == 0:
                print(f"Episode: {episode:4d} | Score: {env.score:2d} | Reward: {episode_reward:6.2f} | Steps: {step_count:3d}")

            if episode % save_interval == 0:
                file_path = f"{model_dir}/ppo_snake_ep{episode}_score{env.score}.pth"
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                torch.save(ppo_agent.policy.state_dict(), file_path)

        # 끝까지 학습이 완료된 경우에만 결과 요약 저장 (중단/오류 시에는 만들지 않음)
        last500 = scores[-500:]
        result = {
            "exp_name": exp_name,
            "episodes": len(scores),
            "total_steps": time_step,
            "last500_mean_score": sum(last500) / len(last500) if last500 else 0.0,
            "total_minutes": (time.time() - start_time) / 60,
        }
        with open(f"{exp_dir}/result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"📊 마지막 {len(last500)}판 평균 점수: {result['last500_mean_score']:.2f} | 학습 시간: {result['total_minutes']:.1f}분")

    except KeyboardInterrupt:
        print("\n🛑 학습이 사용자에 의해 중단되었습니다.")

    finally:
        # 안전한 종료 처리 (강제 종료되더라도 마지막 모델 저장)
        os.makedirs(model_dir, exist_ok=True)
        final_path = f"{model_dir}/ppo_snake_final.pth"
        torch.save(ppo_agent.policy.state_dict(), final_path)
        print(f"현재까지 학습된 모델이 '{final_path}'에 저장되었습니다.")
        writer.close()


if __name__ == '__main__':
    # 기본값은 기존 하이퍼파라미터와 동일합니다. 이어서 학습하려면 --policy_path에 정책 파일 경로를 넣으면 됩니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None, help='기본값: manual_YYYYmmdd_HHMMSS')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--dist_reward', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--update_timestep', type=int, default=2000)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default='')
    parser.add_argument('--allow_existing', action='store_true', help='이미 있는 실험 이름 폴더에도 쓰기 허용 (중단된 실험 재시작용)')
    args = parser.parse_args()
    train(policy_path=args.policy_path, episodes=args.episodes, exp_name=args.exp_name,
          dist_reward=args.dist_reward, lr=args.lr, gamma=args.gamma, epochs=args.epochs,
          eps_clip=args.eps_clip, update_timestep=args.update_timestep,
          entropy_coef=args.entropy_coef, seed=args.seed, allow_existing=args.allow_existing)
