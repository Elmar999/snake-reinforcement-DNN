import cv2
import numpy as np
import tensorflow as tf

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500


class Player:
    def __init__(self):
        self.px = 0
        self.py = 0
        self.done = False
        self.action = None
        self.state = None
        self.reward = 0
        self.food_x = 0
        self.food_y = 0
        self.env = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT), np.uint8)
        self.score = 0
        self.snake_px = []
        self.snake_py = []

    def init_food(self):
        offset = 10
        self.food_x = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        self.food_y = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)

    def init_player(self):
        offset = 0
        self.px = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        self.py = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        # self.snake_px.append(self.px)
        # self.snake_py.append(self.py)

    def move(self, action):
        if action == 0:  # up
            border_point = 0
            if self.py > border_point:
                self.py -= 10
        elif action == 1:  # right
            border_point = WINDOW_HEIGHT
            if self.px < border_point:
                self.px += 10
        elif action == 2:  # down
            border_point = WINDOW_WIDTH
            if self.py < border_point:
                self.py += 10
        elif action == 3:  # left
            border_point = 0
            if self.px > border_point:
                self.px -= 10

    def update_length(self):
        # add one cell to tail
        pass

    def update_positions():
        pass

    def user_play(self, movement):
        dc = dict()
        dc['w'] = 0
        dc['d'] = 1
        dc['s'] = 2
        dc['a'] = 3

        if movement == ord('w'):
            return dc['w']
        elif movement == ord('d'):
            return dc['d']
        elif movement == ord('s'):
            return dc['s']
        elif movement == ord('a'):
            return dc['a']

    def run(self, state, random=False, model_path=False, render=False):
        self.episode_env = self.env.copy()
        state = [self.px, self.py, self.food_x, self.food_y]

        if render:
            cv2.circle(self.episode_env, (self.food_x, self.food_y),
                       5, (255, 255, 255), 5)
            cv2.circle(self.episode_env, (self.px, self.py),
                       5, (255, 255, 255), 1)
            cv2.putText(self.episode_env, "score" + str(self.score),
                        (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 1)
            cv2.imshow("env", self.episode_env)
            if random:
                cv2.waitKey(1)
            else:
                movement = cv2.waitKey(0)
                self.action = self.user_play(movement)
                print(self.action)

        if self.action == 3:
            self.move(3)
        elif self.action == 1:
            self.move(1)
        elif self.action == 2:
            self.move(2)
        elif self.action == 0:
            self.move(0)

        state = [self.px, self.py, self.food_x, self.food_y]

        player_distance = np.array([self.px, self.py])
        food_distance = np.array([self.food_x, self.food_y])
        distance = np.linalg.norm(np.subtract(player_distance, food_distance))
        if distance < 12:
            self.init_food()
            self.score += 1
            print(self.score)
            self.update_length()

        if self.px <= 0 + 10 or self.py <= 0 + 10 or self.px >= WINDOW_WIDTH - 10 or self.py >= WINDOW_HEIGHT - 10:
            self.done = True
            self.reward = -10
            print("done")
        else:
            self.done = False

        return state, self.action, distance, self.score, self.done

    def run_v2(self, state, model=None):
        if model:
            self.action = np.argmax(
                model.predict(np.reshape(state, (1, 4))))
            # print(self.action)
            self.actions_history.append(self.action)

            if len(self.actions_history) > 4:
                dc = dict()
                history = self.actions_history[-4:]
                for i in history:
                    dc[i] = 1
                if len(dc) == 2:
                    self.action = np.random.randint(0, 4)

        if self.action == 3:
            self.move(3)
        elif self.action == 1:
            self.move(1)
        elif self.action == 2:
            self.move(2)
        elif self.action == 0:
            self.move(0)

        # self.state = [self.px, self.py, self.food_x, self.food_y]

        player_distance = np.array([self.px, self.py])
        food_distance = np.array([self.food_x, self.food_y])
        distance = np.linalg.norm(np.subtract(player_distance, food_distance))
        if distance < 12:
            self.init_food()
            self.score += 1
            self.update_length()
            # print(self.score)

        if self.px <= 0 or self.py <= 0 or self.px >= WINDOW_WIDTH or self.py >= WINDOW_HEIGHT:
            self.done = True
            self.reward = -10
            print(self.px, self.py, self.food_x, self.food_y)
            print("done")
        else:
            self.done = False
