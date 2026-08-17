from buffer import ReplayBuffer
from model import QNet
import torch
from torch import nn
from torch.optim import Adam
import numpy as np

class TaxiAgentDQN:

    def __init__(self, env, inDim, outDim, bufferCapacity, seed=None, double=False):

        self.double = double

        self.env = env

        self.qnet = QNet(inDim, outDim)
        self.targetNet = QNet(inDim, outDim)

        self.targetNet.load_state_dict(self.qnet.state_dict())
        self.targetNet.eval()

        self.optim = Adam(self.qnet.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

        self.buffer = ReplayBuffer(capacity=bufferCapacity, seed=seed)

        self.gamma = 1.0
        self.eps = 1.0
        self.batchSize = 64

        self.rng = np.random.default_rng(seed=seed)

        self.legalMaskMatrix = torch.Tensor([self.env.get_legal_actions_mask(i) for i in range(env.nZones)])

        self.pArr = 1 - np.exp(-(env.demands/env.nDays)/env.kappa)

        self.pNbrMean = np.zeros((env.nZones, self.pArr.shape[1]))
        self.pNbrMax = np.zeros((env.nZones, self.pArr.shape[1]))
        for idx, i in enumerate(self.pArr):
            nbr = env.adjacentZones[idx]
            if len(nbr) > 0:
                self.pNbrMean[idx] = self.pArr[nbr, :].mean(axis=0)
                self.pNbrMax[idx] = self.pArr[nbr, :].max(axis=0)

    # featurize the states (N,2) -> (N,5)
    def featurize_states(self, states):
        
        zone, bucket = states[:,0], states[:,1]

        coords = self.env.centroids[zone]
        cx, cy = coords[:,0], coords[:,1]

        angle = (2*np.pi * bucket)/288
        
        cosT = np.cos(angle)
        sinT = np.sin(angle)

        elapsedT = bucket/288

        pCur = self.pArr[zone, bucket]
        pNbrMean = self.pNbrMean[zone, bucket]
        pNbrMax = self.pNbrMax[zone, bucket]

        return np.stack([cx, cy, cosT, sinT, elapsedT, pCur, pNbrMean, pNbrMax], axis = 1).astype(np.float32)

    def select_action(self, state):

        zone, _ = state

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
        
        stateFeature = torch.from_numpy(self.featurize_states(np.array([state])))

        with torch.no_grad():
            action = self.qnet(stateFeature).squeeze(0)
        
        action[illegal] = float('-inf')

        return int(torch.argmax(action).item())
    
    def greedy_policy(self, state):

        zone, bucket = state

        legalActionMask = self.env.get_legal_actions_mask(zone)

        illegal = []

        for i in range(len(legalActionMask)):
            if legalActionMask[i] == 0:
                illegal.append(i)
        
        stateFeature = torch.from_numpy(self.featurize_states(np.array([state])))
        
        with torch.no_grad():
            action = self.qnet(stateFeature).squeeze(0)
        
        action[illegal] = float('-inf')

        return int(torch.argmax(action).item())
    
    def random_policy(self, state):

        zone, _ = state

        legalActionMask = self.env.get_legal_actions_mask(zone)
        legal = [i for i, m in enumerate(legalActionMask) if m == 1]
        
        return self.rng.choice(legal)

    def decay_eps(self, episode, decay_episodes):
        frac = min(1.0, episode / decay_episodes)
        self.eps = 1.0 + frac * (0.05 - 1.0)

    def sync_target(self):
        self.targetNet.load_state_dict(self.qnet.state_dict())
    
    def learn(self):

        if len(self.buffer) < self.batchSize:
            return
        
        states, actions, rewards, nextStates, dones = self.buffer.sample(self.batchSize)
        
        featureStates = self.featurize_states(states)
        featureNextStates = self.featurize_states(nextStates)
        actionsT = torch.as_tensor(actions, dtype=torch.long).view(-1, 1)
        rewardsT = torch.as_tensor(rewards, dtype=torch.float32)
        donesT = torch.as_tensor(dones, dtype=torch.float32)

        qPred = self.qnet(torch.from_numpy(featureStates)).gather(1, actionsT)

        with torch.no_grad():
            
            nextZones = torch.as_tensor(nextStates[:,0], dtype=torch.long)
            batchMask = self.legalMaskMatrix[nextZones]
            
            if self.double:
                qOnlineNext = self.qnet(torch.from_numpy(featureNextStates))
                qOnlineNext[batchMask==0] = float('-inf')
                nextAction = qOnlineNext.argmax(dim=1, keepdim=True)
                maxNext = self.targetNet(torch.from_numpy(featureNextStates)).gather(1, nextAction)
            else:
                qNext = self.targetNet(torch.from_numpy(featureNextStates))
                qNext[batchMask==0] = float('-inf')
                maxNext = qNext.max(dim = 1).values.view(-1,1)
            
            target = rewardsT + self.gamma * maxNext * (1 - donesT)
            
        loss = self.loss(qPred, target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optim.step()
    
    def predict_value(self, state):

        zone, _ = state

        legalActionMask = self.env.get_legal_actions_mask(zone)

        illegal = []

        for i in range(len(legalActionMask)):
            if legalActionMask[i] == 0:
                illegal.append(i)
        
        stateFeature = torch.from_numpy(self.featurize_states(np.array([state])))
        
        with torch.no_grad():
            value = self.qnet(stateFeature).squeeze(0)
        
        value[illegal] = float('-inf')

        return float(torch.max(value).item())