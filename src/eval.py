from qLearning_agent import TaxiAgent
from dqn_agent import TaxiAgentDQN
from env import TaxiEnv
import numpy as np
import torch


def one_evaluate(env:TaxiEnv, policyFn, seed):

    state = env.reset(seed = seed)
    done = False
    totalReturn = 0.0

    while not done:

        a = policyFn(state)
        nextState, r, done = env.step(a)
        
        state = nextState
        totalReturn += r
    
    return totalReturn

def getStats(arr1, arr2, arr3, pol1Name, pol2Name, pol3Name):

    mean1 = np.mean(arr1)
    median1 = np.median(arr1)
    std1 = np.std(arr1)
    stdError1 = np.std(arr1)/np.sqrt(len(arr1))

    mean2 = np.mean(arr2)
    median2 = np.median(arr2)
    std2 = np.std(arr2)
    stdError2 = np.std(arr2)/np.sqrt(len(arr2))

    mean3 = np.mean(arr3)
    median3 = np.median(arr3)
    std3 = np.std(arr3)
    stdError3 = np.std(arr3)/np.sqrt(len(arr3))

    meanDelta = np.mean([arr1[i] - arr2[i] for i in range(len(arr1))])
    seDelta = np.std([arr1[i] - arr2[i] for i in range(len(arr1))])/(np.sqrt(len(arr1)))

    meanDelta1 = np.mean([arr3[i] - arr1[i] for i in range(len(arr1))])
    seDelta1 = np.std([arr3[i] - arr1[i] for i in range(len(arr1))])/(np.sqrt(len(arr1)))

    meanDelta2 = np.mean([arr3[i] - arr2[i] for i in range(len(arr3))])
    seDelta2 = np.std([arr3[i] - arr2[i] for i in range(len(arr3))])/(np.sqrt(len(arr3)))

    wins = [arr1[i] >= arr2[i] for i in range(len(arr1))]
    winRate = np.mean(wins)

    wins1 = [arr3[i] >= arr1[i] for i in range(len(arr3))]
    winRate1 = np.mean(wins1)

    wins2 = [arr3[i] >= arr2[i] for i in range(len(arr3))]
    winRate2 = np.mean(wins2)

    statStr = f"""

    {pol1Name}:

    mean: {mean1:.1f}
    median: {median1:.1f}
    std: {std1:.1f}
    std error: {stdError1:.1f}

    ------------------------------

    {pol2Name}:

    mean: {mean2:.1f}
    median: {median2:.1f}
    std: {std2:.1f}
    std error: {stdError2:.1f}

    ------------------------------

    {pol3Name}:

    mean: {mean3:.1f}
    median: {median3:.1f}
    std: {std3:.1f}
    std error: {stdError3:.1f}

    ------------------------------

    {pol1Name} - {pol2Name}:

    mean delta: {meanDelta:.1f}
    std error delta: {seDelta:.1f}

    {meanDelta:.1f} +- {seDelta:.1f}    

    win rate: {winRate*100:.1f}%

    ------------------------------

    {pol3Name} - {pol2Name}:

    mean delta: {meanDelta2:.1f}
    std error delta: {seDelta2:.1f}

    {meanDelta2:.1f} +- {seDelta2:.1f}    

    win rate: {winRate2*100:.1f}%

    ------------------------------

    {pol3Name} - {pol1Name}:

    mean delta: {meanDelta1:.1f}
    std error delta: {seDelta1:.1f}

    {meanDelta1:.1f} +- {seDelta1:.1f}    

    win rate: {winRate1*100:.1f}%

    ------------------------------
    """

    return statStr


def run_evaluate(dataPaths, seeds):

    print(f'evaluating for {len(seeds)} random seeds...')

    greedyArrs = []
    randomArrs = []
    dqnGreedyArrs = []

    env = TaxiEnv(dataPaths)

    agent = TaxiAgent(env, 0)
    agent.qTable = np.load('../result/qTable.npy', allow_pickle=True)

    agentDQN = TaxiAgentDQN(env, inDim = 8, outDim=69, bufferCapacity=1, seed=0, double=True)
    agentDQN.qnet.load_state_dict(torch.load('../result/qnet_ddqn.pth'))
    agentDQN.qnet.eval()

    for s in seeds:

        agent.rng = np.random.default_rng(s)
        agentDQN.rng = np.random.default_rng(s)

        greedyPolicyFn = agent.greedy_policy
        randomPolicyFn = agent.random_policy
        dqnGreedyPolicyFn = agentDQN.greedy_policy

        greedyTotal = one_evaluate(env, greedyPolicyFn, s)
        randomTotal = one_evaluate(env, randomPolicyFn, s)
        dqnGreeedyTotal = one_evaluate(env, dqnGreedyPolicyFn, s)

        greedyArrs.append(greedyTotal)
        randomArrs.append(randomTotal)
        dqnGreedyArrs.append(dqnGreeedyTotal)

    statsStr = getStats(greedyArrs, randomArrs, dqnGreedyArrs, 'greedyTabular', 'random', 'greedyDoubleDQN')

    print(statsStr)

    dqnAgent  = TaxiAgentDQN(env, 8, 69, 1, seed=0, double=False)
    dqnAgent.qnet.load_state_dict(torch.load('../result/qnet_v2.pth')); dqnAgent.qnet.eval()

    ddqnAgent = TaxiAgentDQN(env, 8, 69, 1, seed=0, double=True)
    ddqnAgent.qnet.load_state_dict(torch.load('../result/qnet_ddqn.pth')); ddqnAgent.qnet.eval()

    print('DQN  :', overestimation_study(env, dqnAgent,  seeds))
    print('DDQN :', overestimation_study(env, ddqnAgent, seeds))



def overestimation_study(env, agent, seeds):
    predicted, realized = [], []
    for s in seeds:
        s0 = env.reset(seed=s)
        q0 = agent.predict_value(s0)              # prediction BEFORE any stepping
        G  = one_evaluate(env, agent.greedy_policy, s)   # greedy rollout on the SAME seed → realized
        predicted.append(q0)
        realized.append(G)
    predicted, realized = np.array(predicted), np.array(realized)
    bias = predicted - realized
    return {
        'meanPredicted': float(predicted.mean()),
        'meanRealized':  float(realized.mean()),
        'meanBias':      float(bias.mean()),
        'seBias':        float(bias.std() / np.sqrt(len(bias))),
    }

if __name__ == '__main__':

    dataPaths = {
    'demands': '../data/artifacts/demand.npy',
    'destinationProbab': '../data/artifacts/destinationProbab.npy',
    'fareDuration': '../data/artifacts/fare_duration.npy',
    'id2idx': '../data/artifacts/id2idx.json',
    'idx2id': '../data/artifacts/idx2id.json',
    'adjacentZones': '../data/artifacts/adjacent_zones.json',
    'centroid': '../data/artifacts/centroids.npy'
    }
    
    seeds = range(1000)

    run_evaluate(dataPaths, seeds)

