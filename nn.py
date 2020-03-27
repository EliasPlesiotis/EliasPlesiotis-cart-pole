from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np


class Model:
    def __init__(self):
        # The model
        self.model = Sequential()
        self.model.add(Dense(16, input_shape=[4,], activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(2, activation='linear'))

        self.model.compile(optimizer=Adam(0.001), 
                           loss='mse')

        # Load the model if available
        try:
            self.model.load_weights('pole.h5')
        except:
            pass

        # The memory
        self.mem_size = 10000
        self.batch_size = 32
        self.index = 0

        self.states = np.zeros([self.mem_size, 4])
        self.states_ = np.zeros([self.mem_size, 4])
        self.actions = np.zeros([self.mem_size])
        self.rewards = np.zeros([self.mem_size])
        self.done = np.zeros([self.mem_size])

        # Greedy behavior
        self.e = 1
        self.e_decay = 0.0001

    def enough_mem(self):
        return self.index > self.batch_size

    def remember(self, state, reward, action, done, state_):
        self.states[self.index%self.mem_size] = state
        self.states_[self.index%self.mem_size] = state_
        self.actions[self.index%self.mem_size] = action
        self.rewards[self.index%self.mem_size] = reward
        self.done[self.index%self.mem_size] = done
        self.index += 1

    def save(self):
        self.model.save('pole.h5')
    
    def predict(self, state):
        if np.random.random() < self.e:
            return int(np.random.choice([0, 1]))
        
        out = self.model.predict(np.array([state]))
        return 1 if out[0][1] > out[0][0] else 0

    def train(self):
        if not self.enough_mem():
            return

        self.e -= self.e_decay
        self.e = 0.01 if self.e < 0.01 else self.e
        
        indexs = np.random.randint(0, self.mem_size, size=[self.batch_size])

        sample_s = [ np.array([self.states[i]]) for i in indexs ]
        sample_s_ = [ np.array([self.states_[i]]) for i in indexs ]
        sample_act = [ int(self.actions[i]) for i in indexs ]
        sample_r = [ self.rewards[i] for i in indexs ]
        sample_d = [ int(self.done[i]) for i in indexs ]

        X = [ s[0] for s in sample_s ]
        Y = []

        for state, state_, action, reward, done in zip(sample_s, sample_s_, sample_act, sample_r, sample_d):
            current_q = self.model.predict(state)[0]
            next_q = self.model.predict(state_)[0]
            max_q = np.amax(next_q)

            current_q[action] = reward + max_q * .99 * ( 1 - done )

            Y.append(current_q)
            
        X, Y = np.array(X), np.array(Y)

        self.model.fit(X, Y, epochs=1, verbose=0)