from env import *
import numpy as np
from collections import deque
import tensorflow as tf
import random
import time
from tqdm import tqdm


class Game:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.state = np.array((0, 0, 0, 0)).reshape((1, 4))
        self.epislon = 1.0
        self.epislon_min = 0.01
        self.epislon_decay = 0.995
        self.gamma = 0.95
        self.t1 = time.time()
        self.load_model = False
        self.episode = 0
        self.player = Player()

    def act(self, state, model):
        if np.random.rand() <= self.epislon:
            return np.random.choice([0, 1, 2, 3])
        act_values = model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def random_game(self, random=True):
        # if random is false means we will play game
        score_req = -200
        training_data = []
        accepted_scores = []

        for game_index in tqdm(range(100)):
            # reset environment
            
            player = Player()
            player.init_player()
            player.init_food()
            score = 0
            game_memory = []
            game_prev_score = 0
            previous_observation = []
            prev_distance = 1000
            for step_index in range(1000):
                if random:
                    action = np.random.randint(0, 4)  # make random choice
                    player.move(action)
                    
                state = [player.px, player.py, player.food_x, player.food_y]
                state, action, distance, game_score, _ = player.run(
                    state, random=False, render=True)

                if len(previous_observation) > 0:
                    game_memory.append([previous_observation, action])
                previous_observation = state

                # if distance < prev_distance:
                #     prev_distance = distance
                #     reward = 10
                if game_score != 0:
                    game_prev_score = game_score
                    reward = 100
                    # print(game_score, score)
                    break
                else:
                    reward = -5
                score += reward
                print("end")

                # print(score)
            if score >= score_req:
                accepted_scores.append(score)
                for data in game_memory:
                    if data[1] == 0:
                        output = [1, 0, 0, 0]
                    elif data[1] == 1:
                        output = [0, 1, 0, 0]
                    elif data[1] == 2:
                        output = [0, 0, 1, 0]
                    elif data[1] == 3:
                        output = [0, 0, 0, 1]
                    training_data.append([data[0], output])

        training_data_save = np.array(training_data)
        np.save('/home/elmar/Documents/projects/rl_learning/snake_game/snake_data.npy',
                training_data_save, allow_pickle=True)
        print(accepted_scores)
        return training_data

    def game_run(self):
        player = Player()
        player.init_food()
        player.init_player()

        while True:
            state = [player.px, player.py, player.food_x, player.food_y]
            _, _, _, _, done = player.run(
                state, model_path='/home/elmar/Documents/projects/rl_learning/snake_game/snake_model.h5')
            if done:
                player.init_food()
                player.init_player()


def train_model(training_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001))

    X = np.array([i[0] for i in training_data]
                 ).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]
                 ).reshape(-1, len(training_data[0][1]))
    model.fit(X, y, epochs=100)
    model.save(
        "/home/elmar/Documents/projects/rl_learning/snake_game/snake_model.h5")
    return model


if __name__ == "__main__":

    game = Game()
    game.game_run()

    # training_data = game.train_run_game()
    # training_data = game.random_game(random=False)
    # training_data = np.load('/home/elmar/Documents/projects/rl_learning/snake_game/snake_data.npy')
    # print(ar)
    # model = train_model(training_data)
