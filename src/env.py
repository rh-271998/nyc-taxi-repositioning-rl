import numpy as np
import json

class TaxiEnv:

    def __init__(self, dataPaths):

        self.demands = np.load(dataPaths['demands'], allow_pickle=True)
        self.destinationProbab = np.load(dataPaths['destinationProbab'], allow_pickle=True)
        self.fareDuration = np.load(dataPaths['fareDuration'], allow_pickle=True)
        self.centroids = np.load(dataPaths['centroid'], allow_pickle=True)

        with open(dataPaths['id2idx'], 'r') as f:
            self.id2idx = json.load(f)

        with open(dataPaths['idx2id'], 'r') as f:
            self.idx2id = json.load(f)

        with open(dataPaths['adjacentZones'], 'r') as f:
            self.adjacentZones = json.load(f)

        convertedZones = {}

        for k in self.adjacentZones:
            convertedZones[self.id2idx[str(k)]] = [self.id2idx[str(n)] for n in self.adjacentZones[k]]
        self.adjacentZones = convertedZones

        self.kappa = 5
        self.nDays = 30
        self.bucketMin = 5
        self.dayMin = 1440

        self.nZones = np.shape(self.demands)[0]

        self.clock = None
        self.zone = None
        self.rng = None
        
    @property
    def get_bucket(self):
        return int(self.clock // self.bucketMin)

    def reset(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.clock = 0.0

        self.zone = self.rng.integers(0, self.nZones)
        bucket = self.get_bucket

        return (self.zone, bucket)


    def get_legal_actions_mask(self, zone):

        mask = [0] * self.nZones
        neighbors = self.adjacentZones[zone]

        for n in neighbors:
            mask[n] = 1        
        mask[zone] = 1

        return mask

    def step(self, action):
        
        assert self.get_legal_actions_mask(self.zone)[action] == 1, f"current zone {self.zone} does NOT have a neighbor {action} passed by the agent. please check agent."

        done = False

        bucket = self.get_bucket

        # if action == stay
        if action == self.zone:
            moveCost = 0.0

            demand = self.demands[self.zone, bucket] / self.nDays

            p = 1 - np.exp(-demand/self.kappa)

            if self.rng.random() < p:
                dests, probabs = self.destinationProbab[self.zone, bucket]
                destination = int(self.rng.choice(dests, p=probabs))
                fare, duration = self.fareDuration[self.zone, destination]

                reward = fare - moveCost
                self.clock += duration

                self.zone = destination
            
            else:
                reward = 0.0 - moveCost
                self.clock += self.bucketMin

        else:
            moveCost = 0.0
            _, duration = self.fareDuration[self.zone, action]
            reward = 0.0 - moveCost

            self.clock += duration
            self.zone = action

        if self.clock >= self.dayMin:
            done = True

        bucket = min(self.get_bucket, self.demands.shape[1] - 1)

        return (self.zone, bucket), reward, done




