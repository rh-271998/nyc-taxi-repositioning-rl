"""
Repositioning flow map: Manhattan zones colored by pickup probability (demand),
with arrows showing where the greedy Double DQN policy sends idle drivers,
across four times of day.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch

from env import TaxiEnv
from dqn_agent import TaxiAgentDQN

DATA = '../data'
ART = f'{DATA}/artifacts'
dataPaths = {
    'demands': f'{ART}/demand.npy', 'destinationProbab': f'{ART}/destinationProbab.npy',
    'fareDuration': f'{ART}/fare_duration.npy', 'id2idx': f'{ART}/id2idx.json',
    'idx2id': f'{ART}/idx2id.json', 'adjacentZones': f'{ART}/adjacent_zones.json',
    'centroid': f'{ART}/centroids.npy',
}

# ---- geometry (real projected centroids for plotting) ----
gdf = gpd.read_file(f'{DATA}/taxi_zones/taxi_zones.shp')
zlook = pd.read_csv(f'{DATA}/taxi_zone_lookup.csv')
manIds = zlook[zlook['Borough'] == 'Manhattan']['LocationID']
id2idx = {int(k): v for k, v in json.load(open(f'{ART}/id2idx.json')).items()}
mg = gdf[gdf['LocationID'].isin(manIds)].copy()
mg['idx'] = mg['LocationID'].map(lambda l: id2idx[int(l)])
mg = mg.sort_values('idx').reset_index(drop=True)
cent = mg.geometry.centroid
realXY = {row.idx: (cent.iloc[i].x, cent.iloc[i].y) for i, row in mg.iterrows()}

# ---- agent (greedy Double DQN) ----
env = TaxiEnv(dataPaths)
agent = TaxiAgentDQN(env, inDim=8, outDim=69, bufferCapacity=1, seed=0, double=True)
agent.qnet.load_state_dict(torch.load('../result/qnet_ddqn.pth'))
agent.qnet.eval()

PANELS = [(36, '3:00 AM'), (96, '8:00 AM (morning rush)'),
          (216, '6:00 PM (evening rush)'), (276, '11:00 PM')]

fig, axes = plt.subplots(2, 2, figsize=(14, 15))
for ax, (bucket, label) in zip(axes.ravel(), PANELS):
    pvals = agent.pArr[mg['idx'].values, bucket]          # demand hotness per zone
    mg['p'] = pvals
    mg.plot(ax=ax, column='p', cmap='YlOrRd', vmin=0, vmax=1,
            edgecolor='#888', linewidth=0.4, legend=False)

    nStay = 0
    for idx in mg['idx'].values:
        a = agent.greedy_policy((idx, bucket))
        x, y = realXY[idx]
        if a == idx:                                       # stay & fish
            ax.plot(x, y, 'o', ms=3, color='#08519c', alpha=0.7)
            nStay += 1
        else:                                              # reposition -> a
            tx, ty = realXY[a]
            ax.annotate('', xy=(tx, ty), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='#08306b',
                                        lw=1.4, alpha=0.85))
    ax.set_title(f'{label}   —   {nStay}/69 zones stay & fish', fontsize=12)
    ax.axis('off')

# shared colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label('Pickup probability  p  (demand hotness)')

fig.suptitle('Where the Double DQN policy sends idle drivers\n'
             'arrows = reposition · dots = stay & fish · color = demand',
             fontsize=15, y=0.98)
fig.savefig('../result/plots/fig_repositioning_map.png', dpi=140, bbox_inches='tight')
print('saved ../result/plots/fig_repositioning_map.png')
