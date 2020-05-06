## snake-like-game-ai-reinforcement

Info: This game is not snake game. The goal is to apply reinforcement techniques to teach the player to detect the position of food and get it. AI for real snake game will be programmed later on.

### Introduction
AI for snake-like game trained from the states of environment using Neural Network. Neural Network is implemented using [Keras](https://keras.io/) functional API, that makes it easy to experiment with different architectures. The input of Neural Network is the state elements of our environment. <br/> We have 4 states in this game:

* Player position over x-axis
* Player position over y-axis
* Food position over x-axis
* Food position over y-axis


### Data
Data has always been one of the most important things in AI related problems. In order to teach the player to make an action in each state, data should be well gathered. In order to have a good data I did not play the game randomly but with user-handling keyboard. So I played the game for about 300 episodes, gave the reward when player eats food and gathered data.  


<img src="./imgs/reinfo_learning.png" style="max-width:100%;">

### Model

Neural Network input is consist of 4 neurons, which are showed above. Since we have 4 choices in each state (up, down, left, right), the number of output neurons is 4. The architecture of the neural network is not complex, I tried with two hidden layers but you can experiment with other architectures.

### Installation 

First of all, you need to check if you have all required packages to launch the program. If not then install requirements.

```sh
$ cd ../snake_ai_reinforcement
$ pip3 install -r requirements.txt
```

Then `cd` into game directory and run `python3 main.py -h` to see usage and options.

```sh
$ cd game
$ python3 main.py -h
```

### Usage
```sh
usage: main.py [-h] [--generate_data] [--episode_number FLAG_NB_EPISODE]
               [--train] [--load_data FLAG_LOAD_DATA]
               [--model_path FLAG_MODEL_PATH] [--run_game]
```

#### Options
* `--generate_data` - user should play the game to generate data
    * `--episode_number NB_EPISODE` - specify number of episodes that the game will be played.
* `--train` - train a model with Neural Network
* `--lad_data load_data_path` - if you have several generated data you can specify the data that you want to train on
* `--run_game` - run the game
    * `--model_path model_path` - specify which model (.h5 file) that game will be used during prediction time


#### Usage example 
```sh
$ python3 main.py --run_game --model_path ../game/snake_model.h5
```


### Game Demo
<img src="./imgs/game.gif" style="max-width:50%;">



