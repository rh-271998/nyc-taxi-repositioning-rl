from env import TaxiEnv

dataPaths = {
    'demands': '../data/artifacts/demand.npy',
    'destinationProbab': '../data/artifacts/destinationProbab.npy',
    'fareDuration': '../data/artifacts/fare_duration.npy',
    'id2idx': '../data/artifacts/id2idx.json',
    'idx2id': '../data/artifacts/idx2id.json',
    'adjacentZones': '../data/artifacts/adjacent_zones.json'
}

seed = 20

env = TaxiEnv(dataPaths, seed)

zone, bucket = env.reset()

print(zone)

done = False

total = 0
step = 0

while not done:
    a = env.rng.choice([i for i,m in enumerate(env.get_legal_actions_mask(env.zone)) if m == 1])
    (zone, bucket), reward, done = env.step(a)
    total += reward
    step += 1

print(total, step)
