from collections import deque
import numpy as np

class ReplayBuffer:

    def __init__(self, capacity, seed = None):

        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

        self.rng = np.random.default_rng(seed = seed)

    def push(self, state, action, reward, nextState, done):
        self.buffer.append((state, action, reward, nextState, done))

    def sample(self, batchSize):
        
        idx = self.rng.choice(len(self.buffer), size=batchSize, replace=False)
        batch = [self.buffer[i] for i in idx]

        states, actions, rewards, nextStates, dones = zip(*batch)
        
        return (np.array(states),
                np.array(actions),
                np.array(rewards).reshape(-1, 1),
                np.array(nextStates),
                np.array(dones).reshape(-1, 1))
    
    def __len__(self):
        return len(self.buffer)
