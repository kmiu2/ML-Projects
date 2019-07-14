import gym
import tensorflow as tf
import numpy as np
from tensorflow import input_layer

env = gym.make('CartPole-v0')
scoreReq = 50
goalSteps = 500
initialGames = 2000
LR = 1e-3  # Learning rate 1*10^-3

def play_with_model(model):
    scores =[]
    choices = []
    print "Play with trained model"

    for game in range(20):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()

        for step in range(goalSteps):
            env.render()

            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict([prev_obs])[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward

            if done:
                break

        scores.append(score)

    print scores

def train_model(training_data):

    x = []
    y = []

    for i in training_data:
        x.append(i[0])
        y.append(i[1])

    print "X: ", x
    print "Y: ", y

    model = neural_net_model(input_size=len(x[0]))
    model.fit(x, y, n_epochs = 5, show_metric=True, run_id='openai_learning')

    return model


def neural_net_model(input_size):
    # Input Layer
    net = tf.input_layer(shape=[None, input_size], name='input')

    # Hidden Layers
    net = tf.fully_connected(net, 128, activation='relu', name="hlayer1")
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 256, activation='relu', name="hlayer2")
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 512, activation='relu', name="hlayer3")
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 256, activation='relu', name="hlayer4")
    net = tf.dropout(net, 0.8)
    net = tf.fully_connected(net, 128, activation='relu', name="hlayer5")
    net = tf.dropout(net, 0.8)

    # Output layer
    net = tf.fully_connected(net, 2, activation='softmax', name="out")
    net = tf.regression(net, learning_rate = LR)

    model = tf.DNN(net, tensorboard_dir='log')

    return model


def initialPopulation():
    trainingData = []
    acceptedScore = []
    print "Playing Random Games..."

    for game in range(initialGames):
        env.reset()

        # record the game
        thisGameMemory = []
        previousObservation = []
        score = 0

        # Play the game
        for x in range(goalSteps):
            # Only render if you want to visualize it
            # env.render()

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward

            # We save every single time we move what is the state before we move
            if x > 0:
                thisGameMemory.append([previousObservation, int(action)])  # keeps the memory of all games

            previousObservation = observation

            if done:
                break  # If you won or lost, end it

        if score > scoreReq:
            # If its a good score save it
            acceptedScore.append(score)

            output =[]
            # format it for a neural network
            for data in thisGameMemory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output == [1, 0]

            trainingData.append([data[0], output])

    print acceptedScore
    print acceptedScore.__len__()

    return trainingData


data = initialPopulation()
model = train_model(data)
play_with_model(model)
# Back Up
# print env.observation_space
# print env.action_space
#
# env.reset()
#
# action = env.action_space.sample()
# print action
#
# observation, reward, done, info = env.step(action)
# print observation
#
# for i in range(1000):
#     env.render()
#
#     env.step(env.action_space.sample())  # taking random action

