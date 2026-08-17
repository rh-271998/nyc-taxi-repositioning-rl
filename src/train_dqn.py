from env import TaxiEnv
from dqn_agent import TaxiAgentDQN
import numpy as np
import torch
from graph import plot_return_curve

dataPaths = {
    'demands': '../data/artifacts/demand.npy',
    'destinationProbab': '../data/artifacts/destinationProbab.npy',
    'fareDuration': '../data/artifacts/fare_duration.npy',
    'id2idx': '../data/artifacts/id2idx.json',
    'idx2id': '../data/artifacts/idx2id.json',
    'adjacentZones': '../data/artifacts/adjacent_zones.json',
    'centroid': '../data/artifacts/centroids.npy'
}

env = TaxiEnv(dataPaths)
agent = TaxiAgentDQN(env, inDim=8, outDim=69, bufferCapacity=100_000, seed = None, double=True)

nEpisodes = 10000
decayEpisodes = int(0.6 * nEpisodes)
learnSteps = 0

returns = []

for e in range(nEpisodes):

    done = False
    state = env.reset(seed = e)

    epReturn = 0.0

    while not done:
        action = agent.select_action(state)
        nextState, reward, done = env.step(action)
        
        agent.buffer.push(state, action, reward, nextState, done)

        agent.learn()
        learnSteps += 1

        if learnSteps % 1000 == 0:
            agent.sync_target()

        epReturn += reward
        state = nextState

    agent.decay_eps(e, decayEpisodes)
    returns.append(epReturn)

    if (e+1) % 500 == 0:
        print(f"ep: {e} | avg return (last 50): {np.mean(returns[-50:]):.1f} | eps: {agent.eps:.2f}")

torch.save(agent.qnet.state_dict(), '../result/qnet_ddqn.pth')

plot_return_curve(returns = returns,
                  savePath='../result/plots/ddqn_return_curve.png',
                  baseline=178.6, window=200,
                  title='Double-DQN: return vs episode')
