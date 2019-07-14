# To filter out the safe to ignore warnings
import warnings
import gym
import random

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

env = gym.make('FrozenLake-v0')

# print "Observation space: ", env.observation_space  # 16 tiles
# print "Action space: ", env.action_space  # Up, Right, Left, Down
# 0 Left, 1 Down, 2 Right, 3 Up

score = 0

for runs in range(10000):
    env.reset()

    for i in range(100):
            observation, reward, done, info = env.step(random.randrange(1, 2))  # moves around map randomly

            # If you reach the goal
            if done:
                # env.render()  # displays map
                score += reward
                break

# print observation
# print reward
# print done
# print info

print "Score: ", score
