from env import *
import numpy as np
from collections import deque
import tensorflow as tf
import random
import time
from tqdm import tqdm
import cProfile
import config


class Game:
    def __init__(self):
        self.state = np.array((0, 0, 0, 0)).reshape((1, 4))

    def generate_data(self, nb_episodes):
        # if random is false means we will play game
        score_requirement = -200
        training_data = []
        accepted_scores = []

        for game_index in tqdm(range(nb_episodes)):
            # reset environment
            player = Player()
            player.init_player()
            player.init_food()
            player.score = 0
            score = 0
            game_memory = []
            game_prev_score = 0
            prev_state = []
            prev_distance = 1000
            for step_index in range(1000):
                state = [player.px, player.py, player.food_x, player.food_y]
                # player.preprocessing()
                new_state, action, game_score = player.preprocessing(state)
                if len(prev_state) > 0:
                    game_memory.append([prev_state, action])
                prev_state = new_state

                if game_score != 0:
                    game_prev_score = game_score
                    reward = 100
                    print(reward)
                    # break
                else:
                    reward = -5
                score += reward

            if score >= score_requirement:
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
        np.save(config.DATA_PATH,
                training_data_save, allow_pickle=True)
        print(accepted_scores)
        return training_data

     def train(self, training_data):
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
        model.fit(X, y, epochs=config.EPOCHS)
        model.save(config.MODEL_PATH)
        return model


    def render(self, player):
        episode_env = player.env.copy()
        cv2.circle(episode_env, (player.food_x, player.food_y),
                   5, (255, 255, 255), 5)
        cv2.circle(episode_env, (player.px, player.py),
                   5, (255, 255, 255), 1)
        cv2.putText(episode_env, "score" + str(player.score),
                    (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 1)
        cv2.imshow("env", episode_env)
        cv2.waitKey(1)

    def game_run(self, model_path=None):

        player = Player()
        player.init_food()
        player.init_player()
        if model_path:
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.models.load_model(config.MODEL_PATH)

        while True:
            # rendering
            self.render(player)

            state = [player.px, player.py, player.food_x, player.food_y]
            player.run_v2(state, model)
            if player.done is True:
                # TODO
                # restart the game after 10 seconds
                return
            time.sleep(0.05)

   
