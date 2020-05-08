import cv2
import numpy as np
import tensorflow as tf

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500


class Player:
    """
    Player class with corresponding functionalities

    Attributes:
        px (int): position of player on x-axis
        py (int): position of player on y-axis
        food_x (int): position of food on x-axis
        food_y (int): position of food on y-axis
        reward (int): give reward in case player eat the food
        score (int): score of the episode
        velocity (int): velocity of player
        done (bool): indicate if episode is done or not
        action (int): player move
        env (np.array): environment of the game
    """

    def __init__(self):
        """
        Initialize Player

        Args:
            env (np.array): environment of the game in the shape of (WINDOW_WIDTH, WINDOW_HEIGHT)
        """
        self.env = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT), np.uint8)
        self.px = None
        self.py = None
        self.velocity = 5
        self.snake_px = []
        self.snake_py = []
        self.food_x = None
        self.food_y = None
        self.reward = None
        self.score = 0
        self.done = False
        self.action = None
        self.actions_history = []

    def init_food(self):
        """
        initialize food position randomly 
        """
        offset = 10
        self.food_x = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        self.food_y = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)

    def init_player(self):
        """
        initialize player position randomly 
        """
        offset = 0
        self.px = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        self.py = np.random.randint(0 + offset, WINDOW_HEIGHT-offset)
        self.snake_px.append(self.px)
        self.snake_py.append(self.py)
    
    def update_length(self, x_change, y_change):
        for i in range(len(self.snake_px)):
            self.snake_px[i] += x_change
            self.snake_py[i] += y_change

    def move(self, action):
        """
        change position of player according to action.
        Args:
            action (arg): action made by user or model
        """
        if action == 0:  # up
            self.py -= self.velocity
            x_change = 0
            y_change = -10
            self.update_length(x_change, y_change)
        
        elif action == 1:  # right
            self.px += self.velocity
            x_change = 10
            y_change = 0
            self.update_length(x_change, y_change)

        elif action == 2:  # down
            self.py += self.velocity
            x_change = 0
            y_change = 10
            self.update_length(x_change, y_change)

        elif action == 3:  # left
            self.px -= self.velocity
            x_change = -10
            y_change = 0
            self.update_length(x_change, y_change)

        return x_change, y_change

    def user_play(self, movement):
        """
        Handling user input from keyboard
        Args:
            movement (char): key pressed by user 
        """
        option = dict()
        option['w'] = 0
        option['d'] = 1
        option['s'] = 2
        option['a'] = 3

        if movement == ord('w'):
            return option['w']
        elif movement == ord('d'):
            return option['d']
        elif movement == ord('s'):
            return option['s']
        elif movement == ord('a'):
            return option['a']
        elif movement == 27:
            exit(0)

    def check_distance(self):
        """
        check distance between player and food
        """
        player_distance = np.array([self.px, self.py])
        food_distance = np.array([self.food_x, self.food_y])
        distance = np.linalg.norm(np.subtract(player_distance, food_distance))
        return distance
    
    def update_snake(self):
        pass

    def preprocessing(self):
        """
        Generate traning data
        """
        self.episode_env = self.env.copy()
        state = [self.px, self.py, self.food_x, self.food_y]

        cv2.circle(self.episode_env, (self.food_x, self.food_y),
                5, (255, 153, 255), 5)
        for i in range(len(self.snake_px)):
            cv2.circle(self.episode_env, (self.snake_px[i], self.snake_py[i]),
                    5, (255, 255, 255), 1)
        
        cv2.putText(self.episode_env, "score" + str(self.score),
                    (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 1)
        cv2.imshow("env", self.episode_env)
        movement = cv2.waitKey(1)                  
        action = self.user_play(movement)

        if action == 3:
            x_change, y_change = self.move(3)
        elif action == 1:
            x_change, y_change = self.move(1)
        elif action == 2:
            x_change, y_change = self.move(2)
        elif action == 0:
            x_change, y_change = self.move(0)


        self.update_snake()        
        new_state = [self.px, self.py, self.food_x, self.food_y]

        distance = self.check_distance()
        if distance < 12:
            self.init_food()
            self.score += 1
            self.snake_px.append(self.px+x_change)
            self.snake_py.append(self.py+y_change)



        if self.px <= 0 or self.py <= 0 or self.px >= WINDOW_WIDTH or self.py >= WINDOW_HEIGHT:
            done = True
            reward = -10
            print("done")

        return new_state, action, self.score

    def run(self, state, model=None):
        """
        run the game
        Args:
            state (np.array): state of the environment based on 4 elements: px, py, food_x, food_y
            model (tf model): model trained on generated data
        """
        if model:
            action = np.argmax(
                model.predict(np.reshape(state, (1, 4))))
            # print(self.action)
            self.actions_history.append(action)

            if len(self.actions_history) > 4:
                dc = dict()
                history = self.actions_history[-4:]
                for i in history:
                    dc[i] = 1
                if len(dc) == 2:
                    action = np.random.randint(0, 4)

        if action == 3:
            self.move(3)
        elif action == 1:
            self.move(1)
        elif action == 2:
            self.move(2)
        elif action == 0:
            self.move(0)

        self.check_distance()

        if self.px <= 0 or self.py <= 0 or self.px >= WINDOW_WIDTH or self.py >= WINDOW_HEIGHT:
            done = True
            self.reward = -10
            print(self.px, self.py, self.food_x, self.food_y)
            print("done")
        else:
            done = False
