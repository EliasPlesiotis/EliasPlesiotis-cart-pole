import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import nn

agent = nn.Model()
agent.e = 0
agent.model.load_weights('pole.h5')

env1 = gym.make('CartPole-v0')
env2 = gym.make('CartPole-v0')

state = env1.reset()
env2.reset()

scores1 = []
scores2 = []

done = False

score1 = 0
score2 = 0

for _ in range(500):
    while not done:
        action = agent.predict(state)
        state, _, done, _ = env1.step(action)
    
        score1 += 1

    done = False
    scores1.append(score1)

    while not done:
        rand_action = np.random.choice([0, 1])
        _, _, done, _ = env2.step(rand_action)
        score2 += 1

    scores2.append(score2)

    done = False

    env1.reset()
    env2.reset()

    score1 = 0
    score2 = 0
    
plt.plot([i for i in range(len(scores1))], scores1)
plt.plot([i for i in range(len(scores2))], scores2)
plt.show()

env1.close()