import numpy as np

class TaxiAgent:

    def __init__(self, env, seed):

        self.rng = np.random.default_rng(seed)

        self.alpha = 0.1
        self.gamma = 1.0

        self.env = env

        self.qTable = np.zeros((env.nZones, env.dayMin//env.bucketMin, env.nZones))

        self.eps = 1.0

    def select_action(self, state):

        zone, bucket = state

        legalActionMask = self.env.get_legal_actions_mask(zone)

        legal = []
        illegal = []

        for i in range(len(legalActionMask)):
            if legalActionMask[i] == 1:
                legal.append(i)
            else:
                illegal.append(i)


        if self.rng.random() < self.eps:
            return self.rng.choice(legal)
        
        qValues = self.qTable[zone, bucket].copy()
        qValues[illegal] = float('-inf')
        
        return np.argmax(qValues)
    
    def greedy_policy(self, state):

        zone, bucket = state

        legalActionMask = self.env.get_legal_actions_mask(zone)

        illegal = []

        for i in range(len(legalActionMask)):
            if legalActionMask[i] == 0:
                illegal.append(i)
        
        qValues = self.qTable[zone, bucket].copy()
        qValues[illegal] = float('-inf')
        
        return np.argmax(qValues)
    
    def random_policy(self, state):

        zone, _ = state

        legalActionMask = self.env.get_legal_actions_mask(zone)
        legal = [i for i, m in enumerate(legalActionMask) if m == 1]
        
        return self.rng.choice(legal)

    def decay_eps(self, episode, decay_episodes):
        frac = min(1.0, episode / decay_episodes)
        self.eps = 1.0 + frac * (0.05 - 1.0)

    def update(self, state, action, reward, nextState, done):

        zone, bucket = state
        nextZone, nextBucket = nextState

        if done:
            target = reward
        
        else:
            legalNextActionMask = self.env.get_legal_actions_mask(nextZone)
            legalNext = [i for i,m in enumerate(legalNextActionMask) if m == 1]
            
            maxNext = max(self.qTable[nextZone, nextBucket, i] for i in legalNext)
            target = reward + self.gamma * maxNext
        
        self.qTable[zone, bucket, action] += self.alpha * (target - self.qTable[zone, bucket, action])

