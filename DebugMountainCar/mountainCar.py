import gym
import random
import numpy as np
import tflearn as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3  # learning rate
env = gym.make('MountainCar-v0')
env.reset()
goal_steps = 500
# score_requirement = 100
initial_games = 10000
position_requirement = 0.5

def random_games():
    for episode in range(10):
        env.reset()
        for t in range(goal_steps):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    max_height = -1.2

    for i in range(initial_games):
        score = 200
        game_memory = []
        prev_observation = []
        position = -0.5

        for i in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)

            # print observation
            # print reward
            # print done
            # print info

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if observation[0] > position:
                position = observation[0]

            if done:
                break

        # Save training data
        if max_height < position:

            max_height = position
            # print max_height

            accepted_scores.append(score)
            output = []

            for data in game_memory:  # data[1] is the choice made 0, 1, or 2 ... data[0] is [position, velocity]
                # print data[0]
                # why [0,1] or [1,0]

                if data[1] == 0:
                    output = [0, 2, 1]
                elif data[1] == 1:
                    output = [1, 2, 0]
                elif data[1] == 2:
                    output = [2, 0, 1]

                training_data.append([data[0], output])

        # print game_memory
        # print score

        # if score >= score_requirement:
        #     accepted_scores.append(score)
        #
        #     output = []
        #
        #     for data in game_memory:
        #         if data[1] == 1:
        #             output = [0, 1]
        #         elif data[1] == 0:
        #             output = [1, 0]
        #
        #         training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    # np.save('saved.npy', training_data_save)

    # print "Average Accepted Score: ", mean(accepted_scores)
    # print "Median Accepted Score: ", median(accepted_scores)
    # print Counter(accepted_scores)

    return training_data


# random_games()
# initial_population()


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    # Hidden Layers
    network = fully_connected(network, 128, activation='relu') # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8) # keep rate 0.8

    network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8)  # keep rate 0.8
    #
    network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    # # network = dropout(network, 0.8)  # keep rate 0.8
    #
    network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8)  # keep rate 0.8

    network = fully_connected(network, 128, activation='relu')  # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8)  # keep rate 0.8

    network = fully_connected(network, 3, activation='softmax')  # this is linked with output array size
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tf.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)  # all your observations
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(x[0]))

    model.fit({'input':x}, {'targets':y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model


training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 200
    game_memory = []
    prev_obs = []
    env.reset()

    for i in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 3)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation

        game_memory.append([new_observation, action])
        score += reward

        if done:
            break

    scores.append(score)

# print "Average Score: ", sum(scores)/len(scores)
# print 'Choice 1: {}, Choice 0 {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices))


