from env import *
import numpy as np
from collections import deque
import tensorflow as tf
import random
import time
from tqdm import tqdm
import cProfile


class Game:
    def __init__(self):
        self.state = np.array((0, 0, 0, 0)).reshape((1, 4))
        self.episode = 0
        self.load_model = False

    def random_game(self):
        # if random is false means we will play game
        score_requirement = -200
        training_data = []
        accepted_scores = []

        for game_index in tqdm(range(1)):
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
                    # print(game_score, score)
                    break
                else:
                    reward = -5
                score += reward
                print("end")

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
        np.save('/home/elmar/Documents/projects/rl_learning/snake_game/snake_data_test.npy',
                training_data_save, allow_pickle=True)
        print(accepted_scores)
        return training_data

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

    def game_run_v2(self):

        player = Player()
        player.init_food()
        player.init_player()
        model = tf.keras.models.load_model(
            '/home/elmar/Documents/projects/rl_learning/snake_game/snake_model.h5')

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


def train_model(training_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(4,), activation='relu'),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001))

    X = np.array([i[0] for i in training_data]
                 ).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]
                 ).reshape(-1, len(training_data[0][1]))
    model.fit(X, y, epochs=1000)
    model.save(
        "/home/elmar/Documents/projects/rl_learning/snake_game/snake_model_best.h5")
    return model


# if __name__ == "__main__":

#     game = Game()
    # game.game_run()
    # game.game_run_v2()

    # training_data = game.train_run_game()
    # training_data = game.random_game()
    # training_data = np.load('/home/elmar/Documents/projects/rl_learning/snake_game/snake_data_200.npy')
    # model = train_model(training_data)
    # print(ar)
    # trained_model = tf.keras.models.load_model('/home/elmar/Documents/projects/rl_learning/snake_game/snake_model_best.h5')
    # trained_model.predict(np.reshape([364, 173, 365, 422], (1, 4)))
    # cProfile.run(game.game_run_v2())
