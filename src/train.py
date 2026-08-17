from env import TaxiEnv
from qLearning_agent import TaxiAgent
from graph import plot_return_curve
import numpy as np


dataPaths = {
    'demands': '../data/artifacts/demand.npy',
    'destinationProbab': '../data/artifacts/destinationProbab.npy',
    'fareDuration': '../data/artifacts/fare_duration.npy',
    'id2idx': '../data/artifacts/id2idx.json',
    'idx2id': '../data/artifacts/idx2id.json',
    'adjacentZones': '../data/artifacts/adjacent_zones.json'
}

seed = 20

env = TaxiEnv(dataPaths)
agent = TaxiAgent(env, seed)

nEpisodes = 50000
decayEpisodes = int(0.6 * nEpisodes)

returns = []

for e in range(nEpisodes):

    state = env.reset(seed=e)
    done = False

    epReturn = 0.0

    while not done:
        a = agent.select_action(state)
        nextState, reward, done = env.step(a)
        agent.update(state, a, reward, nextState, done)

        state = nextState
        epReturn += reward
    
    agent.decay_eps(e, decayEpisodes)

    returns.append(epReturn)

    if (e+1) % 500 == 0:
        print(f"ep: {e} | avg return (last 50): {np.mean(returns[-50:]):.1f} | eps: {agent.eps:.2f}")

np.save('../result/qTable.npy', agent.qTable, allow_pickle=True)
print(f'\nsaved qTable at ../result/qTable.npy\n')

plot_return_curve(returns, baseline=178.6, window=500,
                  title='Tabular Q-learning: return vs episode',
                  savePath='../result/plots/qLearning_curve.png')
