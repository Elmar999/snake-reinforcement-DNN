import cv2
import numpy as np
import tensorflow as tf

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (255, 0, 0)
BLUE_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)


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
        self.env = np.ones((WINDOW_WIDTH, WINDOW_HEIGHT, 3), np.uint8) * 255
        self.grid = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT), np.uint8)
        self.px = None
        self.py = None
        self.velocity = 10
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

    def move(self, action):
        food = False
        """
        change position of player according to action.
        Args:
            action (arg): action made by user or model
        """
        if action == 0:  # up
            self.py -= self.velocity
            x_change = 0
            y_change = -10

        elif action == 1:  # right
            self.px += self.velocity
            x_change = 10
            y_change = 0

        elif action == 2:  # down
            self.py += self.velocity
            x_change = 0
            y_change = 10

        elif action == 3:  # left
            self.px -= self.velocity
            x_change = -10
            y_change = 0

        distance = self.check_distance()
        if distance < 15:
            self.init_food()
            self.score += 1
            food = True

        return food

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
        player_distance = np.array([self.snake_px[-1], self.snake_py[-1]])
        food_distance = np.array([self.food_x, self.food_y])
        distance = np.linalg.norm(np.subtract(player_distance, food_distance))
        return distance

    def check_death(self):
        done = False
        if self.px <= 5 or self.py <= 5 or self.px >= WINDOW_WIDTH - 5 or self.py >= WINDOW_HEIGHT - 5:
            done = True
            reward = -10
            print("done boundaries")
            return done

        else:
            player_distance = np.array([self.snake_px[-1], self.snake_py[-1]])
            for i in range(len(self.snake_px) - 1):
                tail = np.array([self.snake_px[i], self.snake_py[i]])
                distance = np.linalg.norm(np.subtract(player_distance, tail))
                if distance < 1:
                    print("done")
                    done = True
                    return done
        return done

    def update_snake(self, food):
        if food is not True:
            self.grid = self.grid * 0
            for i in range(0, len(self.snake_px)-1):
                self.snake_px[i] = self.snake_px[i+1]
                self.snake_py[i] = self.snake_py[i+1]
                self.grid[self.snake_px[i]][self.snake_py[i]] = 1
            self.snake_px[-1] = self.px
            self.snake_py[-1] = self.py
            self.grid[self.snake_px[-1]][self.snake_py[-1]] = 1

        else:
            self.snake_px.append(self.px)
            self.snake_py.append(self.py)
            self.grid[self.px, self.py] = 1

    def check_neighbours(self):
        """
        check 4 neighbours of snake head: up, down, right, left
        """
        neighbours = [0, 0, 0, 0]
        if self.grid[self.px][self.py - 10] == 1:
            neighbours[0] = 1
        if self.grid[self.px][self.py + 10] == 1:
            neighbours[1] = 1
        if self.grid[self.px + 10][self.py] == 1:
            neighbours[2] = 1
        if self.grid[self.px - 10][self.py] == 1:
            neighbours[3] = 1

        return neighbours

    def preprocessing(self):
        """
        Generate traning data
        """
        self.episode_env = self.env.copy()
        state = [self.px, self.py, self.food_x, self.food_y]

        cv2.circle(self.episode_env, (self.food_x, self.food_y),
                   5, RED_COLOR, 5)

        cv2.circle(self.episode_env, (self.snake_px[-1], self.snake_py[-1]),
                   5, BLUE_COLOR, -1)
        for i in range(0, len(self.snake_px)-1):
            cv2.circle(self.episode_env, (self.snake_px[i], self.snake_py[i]),
                       5, BLACK_COLOR, -1)

        cv2.putText(self.episode_env, "score" + str(self.score),
                    (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, BLACK_COLOR, 1)
        cv2.imshow("env", self.episode_env)
        movement = cv2.waitKey(0)
        action = self.user_play(movement)

        if action == 0:
            food = self.move(0)
            self.update_snake(food)
        elif action == 1:
            food = self.move(1)
            self.update_snake(food)
        elif action == 2:
            food = self.move(2)
            self.update_snake(food)
        elif action == 3:
            food = self.move(3)
            self.update_snake(food)

        new_state = self.check_neighbours() + [self.snake_px[-1], self.snake_py[-1], self.food_x, self.food_y]

        done = self.check_death()

        return new_state, action, done

    def run(self, state, model=None):
        """
        run the game
        Args:
            state (np.array): state of the environment based on 4 elements: px, py, food_x, food_y
            model (tf model): model trained on generated data
        """
        if model:
            action = np.argmax(
                model.predict(np.reshape(state, (1, 8))))
            self.actions_history.append(action)

        if action == 0:
            food = self.move(0)
            self.update_snake(food)
        elif action == 1:
            food = self.move(1)
            self.update_snake(food)
        elif action == 2:
            food = self.move(2)
            self.update_snake(food)
        elif action == 3:
            food = self.move(3)
            self.update_snake(food)

        done = self.check_death()

        return done