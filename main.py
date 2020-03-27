import gym
import math
import matplotlib.pyplot as plt
import numpy as np

import nn


agent = nn.Model()

env = gym.make('CartPole-v0')
state = env.reset()
done = False

scores = []


for i in range(200):
    score = 0
    for j in range(100):
        env.render()
        
        action = agent.predict(state)
        state_, r, done, info = env.step(action)

        if score > 10:
            agent.remember(state, r, action, done, state_)
        agent.train()
        
        score += 1
        state = state_

        if done :
            break

    print('Game '+ str(i) +' Score: '+ str(score) + ' With e: ' + str(agent.e))
    scores.append(score)
    done = False
    score = 0
    env.reset()

            

env.close()

plt.scatter([i for i in range(len(scores))], scores)
plt.show()

agent.model.save('pole.h5')