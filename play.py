import torch
import time
from snake_code import SnakeGame, STATE_DIMS
from PPO_code import PPO

def play_saved_model():
    # 가중치 불러오기
    # TODO: main.py에서 저장했던 모델 파일 경로 적기
    model_path = "best_model/ppo_snake_best.pth"
    weights = torch.load(model_path)

    # 저장된 가중치의 입력 크기로 상태 종류를 판별 (11: basic, 21: body)
    state_mode = next(name for name, dim in STATE_DIMS.items() if dim == weights["actor.0.weight"].shape[1])

    # 뼈대 준비
    state_dim = STATE_DIMS[state_mode]
    action_dim = 3
    env = SnakeGame(state_mode=state_mode)

    # PPO 모델 생성
    ppo_agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, epochs=1, eps_clip=0.2)

    # torch.load로 읽은 딕셔너리를 load_state_dict로 모델에 덮어씌웁니다.
    ppo_agent.policy.load_state_dict(weights)
    
    # 평가 모드 전환
    ppo_agent.policy.eval()
    
    print(f"🎉 {model_path} 로드 완료! 플레이를 시작합니다.")

    state = env.reset()
    
    # 기울기 추적 비활성화
    with torch.no_grad():
        while True:
            env.render()
            
            # 뱀이 너무 빨리 움직이면 사람 눈에 안 보이므로 의도적인 지연 추가
            time.sleep(0.01) 
            
            state_tensor = torch.FloatTensor(state)
            
            # 행동 결정 (Actor 신경망만 사용됨)
            action, _, _ = ppo_agent.policy.act(state_tensor)
            
            state, reward, done = env.step(action)
            
            if done:
                print(f"게임 오버! 뱀 길이(점수): {env.score}")
                break

if __name__ == '__main__':
    play_saved_model()