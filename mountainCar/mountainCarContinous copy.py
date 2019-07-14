import gym
import random
import numpy as np
import tflearn as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3  # learning rate
env = gym.make('MountainCarContinuous-v0')
env.reset()
goal_steps = 2000  # basically now a control to game length
score_requirement = 70  # supposed to be 90
initial_games = 10000
position_requirement = 0.5


def random_games():
    for episode in range(10):
        env.reset()
        for t in range(goal_steps):
            # env.render()

            action = random.randrange(-1,2)

            while action == 0:
                action = random.randrange(-1, 2)

            observation, reward, done, info = env.step([action])

            if done:
                break

# random_games()


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    max_height = -1.2

    for i in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        position = -0.5

        for i in range(goal_steps):
            # env.render()
            # if env.step([0])[0][1] > 0:
            #     action = 1
            # else:
            #     action = -1

            action = random.randrange(-1, 2)  # max force is just 1 and -1

            while action == 0:
                action = random.randrange(-1, 2)

            # print action

            observation, reward, done, info = env.step([action])

            # print observation
            # print reward
            # print done
            # print info

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            # print score

            if observation[0] > position:
                position = observation[0]

            if done:
                break

        # Save training data
        # if 0.477 <= position:  # Try to make sure this only takes in really good ones otherwise it trains on random runs
        #     # env.render()
        #
        #     # max_height = position
        #     # print position
        #
        #     accepted_scores.append(score)
        #     output = []
        #
        #     for data in game_memory:  # data[1] is the choice made 0, 1, or 2 ... data[0] is [position, velocity]
        #         # print data[1]
        #
        #         if data[1] == -1:
        #             output = [-1, 1]
        #         elif data[1] == 1:
        #             output = [1, -1]
        #
        #         training_data.append([data[0], output])

        # print game_memory
        # print score

        if score >= score_requirement:  # Do this instead because these runs actually make it
            # print score

            accepted_scores.append(score)

            output = []

            for data in game_memory:  # Order of these should not matter as the predict will give confidence for each one
                if data[1] == -1:
                    output = [-1, 0, 1]
                elif data[1] == 1:
                    output = [1, 0, -1]
                # elif data[1] == 0:
                #     output == [0, 1, -1]

                # really close if [x, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print "Average Accepted Score: ", mean(accepted_scores)
    print "Median Accepted Score: ", median(accepted_scores)
    print Counter(accepted_scores)

    return training_data

# initial_population()

# the end network should be baesd on velocity

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    # Hidden Layers
    network = fully_connected(network, 128, activation='relu') # 128 nodes, rectified linear (relu)
    network = dropout(network, 0.8) # keep rate 0.8

    # network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8)  # keep rate 0.8
    # #
    network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    network = dropout(network, 0.8)  # keep rate 0.8
    # #
    # network = fully_connected(network, 256, activation='relu')  # 128 nodes, rectified linear (relu)
    # network = dropout(network, 0.8)  # keep rate 0.8

    network = fully_connected(network, 128, activation='relu')  # 128 nodes, rectified linear (relu)
    network = dropout(network, 0.8)  # keep rate 0.8

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
    # model.fit({'input':x}, {'targets':y}, n_epoch=3, snapshot_step=500, run_id='openaistuff')

    return model


training_data = initial_population()
model = train_model(training_data)

# print training_data  # It does have both 1 and -1
print model

scores = []
choices = []

# HOW DO I USE THE MODEL? and training data

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for i in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:

            # print "I get here woohoo"

            action = random.randrange(-1, 2)

            while action == 0:
                action = random.randrange(-1, 2)

        else:  # Problem may be here, print out stuff
            # May need to void the arg max entirely and find a new output value

            # print env.step([0])

            # Using velocity to "solve" ... supposedly this is legit
            # if env.step([0])[0][1] > 0:
            #     action = 1
            # else:
            #     action = -1
            # # print prev_obs

            print "Prev Observation: ", prev_obs.reshape(-1, len(prev_obs), 1)
            # print "Prev Observation 2: ", prev_obs.reshape(-1, len(prev_obs), 1)

            # What does model.predict do?
            # what does the second argument do?

            # Model predict gives you a confidence level of which one to choose from (Output nodes)
            # Because now i have 2 output nodes, argmax has two to choose from versus having 1 node means 1 choice

            print "Model Predict: ", model.predict(prev_obs.reshape(-1, len(prev_obs), 1))

            # ARG MAX WILL NEVER GIVE -1. THEREFORE IT WILL NEVER PUSH LEFT
            # It will forever try to push left and predict oh you should put left which means argmax will always output 0

            if np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))) == 0:
                if random.randrange(1,3) == 2:
                    action = 1
                else:
                    action = -1
            elif np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))) == 1:
                action = -1
            else:
                action = 0

            # closest to logical decision if 1 else -1

            # action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)))

            # This ALWAYS prints [[1.]]

            # therefore this, (getting the index of the largest value in the array, is ALWAYS 0

            # I HAVE CHANGED THIS FROM predict
            # velocity = env.step([0])[0][1]
            # position = env.step([0])[0][0]
            #
            # print velocity
            # print position
            #
            # if abs(velocity) < 0.0001:
            #     if velocity > 0:
            #         action = -1
            #     else:
            #         action = 1
            # else:
            #     if velocity < 0:
            #         action = -1
            #     else:
            #         action = 1
            #     # action = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))
            #
            # print action

            # print "Np: ", np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)))

            # this is zero because np.argmax gives the index of the largest value in the array
            # Because there is only 1 value in the array, the index will always be 0

        choices.append(action)

        # Figure out why action is 0 from np. argmax

        print choices

        # print [np.all([action])]

        new_observation, reward, done, info = env.step([action])
        prev_obs = new_observation

        game_memory.append([new_observation, action])
        score += reward

        if done:
            break

    scores.append(score)

# print "Average Score: ", sum(scores)/len(scores)
# print 'Choice 1: {}, Choice 0 {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices))


