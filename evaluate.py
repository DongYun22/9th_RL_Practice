import argparse
import json
import os
import random

# pygame 환영 메시지가 JSON 출력에 섞이지 않도록 import 전에 숨김
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from snake_code import SnakeGame
from PPO_code import PPO

def evaluate(model_path, games=5, seed=1000):
    # play.py와 동일한 방식으로 모델 생성 및 로드 (렌더링/sleep 없음)
    state_dim = 11
    action_dim = 3
    env = SnakeGame()
    ppo_agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, epochs=1, eps_clip=0.2)
    ppo_agent.policy.load_state_dict(torch.load(model_path))
    ppo_agent.policy.eval()

    scores = []
    with torch.no_grad():
        for i in range(games):
            # 판마다 seed+i로 고정해서 모델 간 비교가 같은 조건에서 이루어지게 함
            random.seed(seed + i)
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)

            state = env.reset()
            while True:
                state_tensor = torch.FloatTensor(state)
                action, _, _ = ppo_agent.policy.act(state_tensor)
                state, reward, done = env.step(action)
                if done:
                    scores.append(env.score)
                    break

    return {
        "model": model_path,
        "games": games,
        "seed": seed,
        "scores": scores,
        "mean": sum(scores) / len(scores),
        "max": max(scores),
        "min": min(scores),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='평가할 .pth 파일 경로')
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--out', type=str, default=None, help='결과 JSON을 저장할 경로 (예: experiments/{exp_name}/eval.json)')
    args = parser.parse_args()

    result = evaluate(args.model, args.games, args.seed)
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
