import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# 색상 RGB 정의
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)      # 뱀의 테두리 색상
BLUE2 = (0, 100, 255)    # 뱀의 안쪽 색상
BLACK = (0, 0, 0)        # 배경 색상

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# 게임 설정값
BLOCK_SIZE = 20
SPEED = 40 # 학습 화면을 볼 때의 속도

class SnakeGame:
    def __init__(self, w=640, h=480, dist_reward=0.1):
        self.w = w
        self.h = h
        self.dist_reward = dist_reward # 사과에 가까워지면 +dist_reward, 아니면 -dist_reward
        # 화면 출력용 (학습 속도를 높이려면 render() 호출을 생략하면 됩니다)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL')
        self.paused = False
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 게임 초기 상태 설정
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
        # 보상 설계를 위한 변수 초기화
        self.frame_iteration = 0
        self.prev_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def step(self, action):
        self._handle_events()
        while self.paused:
            self._handle_events()
            self.clock.tick(15)

        self.frame_iteration += 1
        
        # 에이전트의 행동(Action)에 따라 이동
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = -0.01 # 스텝 패널티 (지연 방지)
        game_over = False

        # 게임 종료 조건 확인 (충돌 또는 무한 루프 아사)
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over

        # 조밀한 보상 (Dense Reward) 적용
        curr_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        if curr_distance < self.prev_distance:
            reward += self.dist_reward
        else:
            reward -= self.dist_reward
        self.prev_distance = curr_distance

        # 사과 획득 확인
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0 # 굶주림 초기화
            # 사과를 먹었으므로 꼬리를 자르지 않음 (길어짐)
        else:
            self.snake.pop() # 사과를 못 먹었으면 꼬리 한 칸 축소시켜 이동 구현
            
        return self.get_state(), reward, game_over

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 벽에 부딪힘
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 자기 몸통에 부딪힘
        if pt in self.snake[1:]:
            return True
        return False
    
    def _move(self, action):
        # action [직진, 우회전, 좌회전]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx] # 직진
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4] # 우회전
        else:
            new_dir = clock_wise[(idx - 1) % 4] # 좌회전

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        # 머리 기준 상하좌우 한 칸의 좌표
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # 위험 감지 (직진, 우회전, 좌회전)
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # 이동 방향
            dir_l, dir_r, dir_u, dir_d,
            
            # 사과 위치
            self.food.x < self.head.x,  # Food left
            self.food.x > self.head.x,  # Food right
            self.food.y < self.head.y,  # Food up
            self.food.y > self.head.y   # Food down
        ]
        return np.array(state, dtype=int)

    def render(self):
        # pygame 이벤트 큐를 비워주어야 창이 '응답없음' 상태에 빠지지 않습니다.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 배경을 검은색으로 지우기
        self.display.fill(BLACK)
        
        # 뱀 그리기
        for pt in self.snake:
            # 뱀의 몸통 (바깥쪽 꽉 찬 사각형)
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # 뱀의 몸통 안쪽 (입체감을 위해 살짝 작은 사각형을 덧그림)
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # 사과 그리기
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 좌측 상단에 현재 점수 표시
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # 화면 업데이트 및 재생 속도 조절
        pygame.display.flip()
        self.clock.tick(SPEED) # __init__ 외부에서 정의한 SPEED(예: 40)에 맞춰 프레임 고정    
