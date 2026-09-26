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

# 상태 종류별 차원. basic: 기존 11개 / body: basic + 몸통 정보 10개
STATE_DIMS = {"basic": 11, "body": 21}
CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
DIR_DELTA = {Direction.RIGHT: (1, 0), Direction.DOWN: (0, 1), Direction.LEFT: (-1, 0), Direction.UP: (0, -1)}

class SnakeGame:
    def __init__(self, w=640, h=480, dist_reward=0.1, state_mode="basic"):
        self.w = w
        self.h = h
        self.dist_reward = dist_reward # 사과에 가까워지면 +dist_reward, 아니면 -dist_reward
        if state_mode not in STATE_DIMS:
            raise ValueError(f"state_mode 는 {list(STATE_DIMS)} 중 하나여야 합니다: {state_mode}")
        self.state_mode = state_mode
        self.state_dim = STATE_DIMS[state_mode]
        # 몸통 정보 계산용 격자 비트보드 (가로 한 칸을 더 둬서 좌우 이동이 다음 줄로 넘어가지 않게 함)
        self.cols = w // BLOCK_SIZE
        self.rows = h // BLOCK_SIZE
        self.stride = self.cols + 1
        self.all_cells = sum(1 << (y * self.stride + x) for y in range(self.rows) for x in range(self.cols))
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
        if self.state_mode == "basic":
            return np.array(state, dtype=int)
        return np.concatenate([np.array(state, dtype=np.float32), self._body_features()])

    def _reachable(self, seed, free, need):
        # seed 칸에서 상하좌우로 퍼져 나가며 도달 가능한 빈 칸 수 (need 칸 이상이면 거기서 멈춤)
        reach = seed
        while True:
            grown = (reach | (reach << 1) | (reach >> 1) | (reach << self.stride) | (reach >> self.stride)) & free
            if grown == reach:
                break
            reach = grown
            if reach.bit_count() >= need:
                break
        return min(reach.bit_count(), need)

    def _body_features(self):
        # 몸통 정보 10개: [직진/우회전/좌회전 방향의 여유 공간 3, 장애물까지 거리 3, 꼬리 위치(좌/우/상/하) 4]
        head = self.snake[0]
        cx, cy = int(head.x // BLOCK_SIZE), int(head.y // BLOCK_SIZE)
        if not (0 <= cx < self.cols and 0 <= cy < self.rows):
            return np.zeros(10, dtype=np.float32) # 벽 밖으로 나가 게임이 끝난 직후의 상태

        cells = [(int(p.x // BLOCK_SIZE), int(p.y // BLOCK_SIZE)) for p in self.snake]
        blocked = set(cells[1:]) # 충돌 판정과 같게 꼬리 칸도 막힌 칸으로 본다
        occupied = 0             # 곧 비워질 꼬리를 뺀 나머지 몸통(머리 포함)의 비트보드
        for x, y in cells[:-1]:
            occupied |= 1 << (y * self.stride + x)
        free = self.all_cells & ~occupied
        length = len(cells)

        idx = CLOCK_WISE.index(self.direction)
        space, dist = [], []
        for turn in (0, 1, -1): # 직진, 우회전, 좌회전 (action 0, 1, 2 와 같은 순서)
            dx, dy = DIR_DELTA[CLOCK_WISE[(idx + turn) % 4]]
            x, y, steps = cx + dx, cy + dy, 0
            while 0 <= x < self.cols and 0 <= y < self.rows and (x, y) not in blocked:
                steps += 1
                x, y = x + dx, y + dy
            dist.append(steps / max(self.cols, self.rows))
            if steps == 0:
                space.append(0.0) # 바로 앞이 벽/몸통
            else:
                seed = 1 << ((cy + dy) * self.stride + (cx + dx))
                space.append(self._reachable(seed, free, length) / length) # 뱀 길이만큼 들어갈 공간이면 1

        tail = self.snake[-1]
        tail_pos = [tail.x < head.x, tail.x > head.x, tail.y < head.y, tail.y > head.y]
        return np.array(space + dist + tail_pos, dtype=np.float32)

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
