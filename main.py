import argparse
from game import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='appsubmit')
    parser.add_argument('--generate_data', help='generate training data',
                        action='store_true', dest='flag_gen_data')
    parser.add_argument(
        '--episode_number', help='number of episodes during training', type=int, dest='flag_nb_episode')
    parser.add_argument('--train', help='train model',
                        action='store_true', dest='flag_train')
    parser.add_argument('--load_data', help='data path',
                        type=str, dest='flag_load_data')
    parser.add_argument('--model_path', help='model path',
                        type=str, default=None, dest='flag_model_path')
    parser.add_argument('--run_game', help='run game',
                        action='store_true', dest='flag_run_game')

    args = parser.parse_args()
    game = Game()

    if args.flag_gen_data is True:
        assert args.flag_nb_episode is not None
        nb_episodes = args.flag_nb_episode
        game.generate_data(nb_episodes)

    if args.flag_train is True:
        if args.flag_gen_data is not True:
            assert args.flag_load_data is not None
            data_path = args.flag_load_data
            game.train(data_path)
        else:
            game.train()

    if args.flag_run_game is True:
        if args.flag_train is not True:
            assert args.flag_model_path is not None
            model_path = args.flag_model_path
            game.game_run(model_path)
        else:
            game.game_run()
