"""
Animated 'day in the life' of the greedy Double DQN driver.
Left: Manhattan zones recolor by demand (pickup prob) as the clock advances; a car
marker moves, styled by mode (searching empty / repositioning empty / carrying pax);
a short trail follows it. Right: cumulative earnings growing through the day.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from env import TaxiEnv
from dqn_agent import TaxiAgentDQN

DATA, ART = '../data', '../data/artifacts'
dataPaths = {
    'demands': f'{ART}/demand.npy', 'destinationProbab': f'{ART}/destinationProbab.npy',
    'fareDuration': f'{ART}/fare_duration.npy', 'id2idx': f'{ART}/id2idx.json',
    'idx2id': f'{ART}/idx2id.json', 'adjacentZones': f'{ART}/adjacent_zones.json',
    'centroid': f'{ART}/centroids.npy',
}

SEED = 7
START_LOCID = 230          # Times Sq / Theatre District — iconic midnight start
FRAME_DT = 6               # minutes of sim time per frame
FPS = 24
MODE_STYLE = {             # marker, color, label
    'searching':     ('o', '#377eb8', 'searching (empty)'),
    'repositioning': ('>', '#ff7f00', 'repositioning (empty)'),
    'carrying':      ('*', '#4daf4a', 'carrying passenger'),
}

# ---------- geometry ----------
gdf = gpd.read_file(f'{DATA}/taxi_zones/taxi_zones.shp')
zlook = pd.read_csv(f'{DATA}/taxi_zone_lookup.csv')
manIds = zlook[zlook['Borough'] == 'Manhattan']['LocationID']
id2idx = {int(k): v for k, v in json.load(open(f'{ART}/id2idx.json')).items()}
mg = gdf[gdf['LocationID'].isin(manIds)].copy()
mg['idx'] = mg['LocationID'].map(lambda l: id2idx[int(l)])
mg = mg.sort_values('idx').reset_index(drop=True)
cent = mg.geometry.centroid
CX = {int(r.idx): cent.iloc[i].x for i, r in mg.iterrows()}
CY = {int(r.idx): cent.iloc[i].y for i, r in mg.iterrows()}

# ---------- agent ----------
env = TaxiEnv(dataPaths)
agent = TaxiAgentDQN(env, inDim=8, outDim=69, bufferCapacity=1, seed=0, double=True)
agent.qnet.load_state_dict(torch.load('../result/qnet_ddqn.pth'))
agent.qnet.eval()

# ---------- roll out one instrumented greedy day ----------
env.reset(seed=SEED)
env.zone = id2idx[START_LOCID]
env.clock = 0.0
segments = []
done = False
while not done:
    z0, t0 = env.zone, env.clock
    a = agent.greedy_policy((z0, env.get_bucket))
    (_, _), r, done = env.step(a)
    t1 = env.clock
    if a == z0:
        mode = 'carrying' if r > 0 else 'searching'
    else:
        mode = 'repositioning'
    segments.append(dict(t0=t0, t1=max(t1, t0 + 0.5), zf=z0, zt=env.zone, mode=mode, fare=float(r)))

DAY = 1440
frame_times = np.arange(0, DAY + FRAME_DT, FRAME_DT)
seg_t0 = np.array([s['t0'] for s in segments])
seg_t1 = np.array([s['t1'] for s in segments])
seg_fare = np.array([s['fare'] for s in segments])
total = seg_fare.sum()


def state_at(tf):
    """position (x,y), mode, cumulative earnings at sim-time tf."""
    i = np.searchsorted(seg_t1, tf, side='right')
    i = min(i, len(segments) - 1)
    s = segments[i]
    span = s['t1'] - s['t0']
    frac = 0.0 if span <= 0 else np.clip((tf - s['t0']) / span, 0, 1)
    x = CX[s['zf']] + frac * (CX[s['zt']] - CX[s['zf']])
    y = CY[s['zf']] + frac * (CY[s['zt']] - CY[s['zf']])
    earn = seg_fare[seg_t1 <= tf].sum()
    return x, y, s['mode'], earn


# ---------- figure scaffold ----------
fig = plt.figure(figsize=(12, 8))
axm = fig.add_axes([0.01, 0.08, 0.58, 0.84])   # map
axe = fig.add_axes([0.70, 0.30, 0.27, 0.46])   # earnings
cax = fig.add_axes([0.09, 0.06, 0.40, 0.02])   # horizontal colorbar under map
axm.axis('off')

mg['p'] = agent.pArr[mg['idx'].values, 0]
mg.plot(ax=axm, column='p', cmap='YlOrRd', vmin=0, vmax=1,
        edgecolor='#999', linewidth=0.3)
zone_coll = axm.collections[-1]
zone_coll.set_clim(0, 1)

trail, = axm.plot([], [], '-', color='#333', lw=1.6, alpha=0.5)
car, = axm.plot([], [], 'o', ms=16, mec='black', mew=1.2, zorder=5)
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('demand  (pickup probability)', fontsize=9)
legend_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                         markersize=13, mec='black', label=lab)
                  for m, c, lab in MODE_STYLE.values()]
axm.legend(handles=legend_handles, loc='lower left', frameon=True, fontsize=9)
clock_txt = axm.set_title('', fontsize=15, fontweight='bold')

# earnings panel
earn_line, = axe.plot([], [], color='#4daf4a', lw=2.5)
earn_dot, = axe.plot([], [], 'o', color='#4daf4a', ms=7)
axe.set_xlim(0, 24)
axe.set_ylim(0, total * 1.12 + 1)
axe.set_xticks([0, 6, 12, 18, 24])
axe.set_xlabel('hour of day')
axe.set_ylabel('cumulative earnings ($)')
axe.set_title('Earnings through the day', fontsize=12)
axe.spines[['top', 'right']].set_visible(False)
axe.grid(alpha=0.3)
earn_txt = axe.text(0, 0, '', fontsize=12, fontweight='bold',
                    ha='center', va='bottom', color='#2b8a3e')

TRAIL_N = 14
trail_x, trail_y = [], []


def update(f):
    tf = frame_times[f]
    bucket = min(int(tf // 5), 287)
    zone_coll.set_array(agent.pArr[mg['idx'].values, bucket])

    x, y, mode, earn = state_at(tf)
    m, c, _ = MODE_STYLE[mode]
    car.set_data([x], [y]); car.set_marker(m); car.set_color(c)
    car.set_markersize(20 if mode == 'carrying' else 15)

    trail_x.append(x); trail_y.append(y)
    trail.set_data(trail_x[-TRAIL_N:], trail_y[-TRAIL_N:])

    hh, mm = int(tf // 60) % 24, int(tf % 60)
    label = {'searching': 'searching for a fare',
             'repositioning': 'driving to a hotter zone',
             'carrying': 'carrying a passenger'}[mode]
    clock_txt.set_text(f'{hh:02d}:{mm:02d}   ·   {label}   ·   ${earn:,.0f}')

    ft = frame_times[:f + 1]
    ec = [state_at(t)[3] for t in ft]
    earn_line.set_data(ft / 60, ec)
    earn_dot.set_data([tf / 60], [earn])
    earn_txt.set_position((min(tf / 60, 21), earn + total * 0.05)); earn_txt.set_text(f'${earn:,.0f}')
    return zone_coll, car, trail, earn_line, earn_dot, clock_txt, earn_txt


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        update(len(frame_times) // 2)     # a midday frame
        fig.savefig('../result/plots/animate_test_frame.png', dpi=110)
        print('saved test frame; total day earnings = $%.0f, %d segments' % (total, len(segments)))
    else:
        anim = FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / FPS, blit=False)
        anim.save('../result/plots/driver_day.gif', writer=PillowWriter(fps=FPS), dpi=80)
        print('saved ../result/plots/driver_day.gif  (%d frames, $%.0f day)' % (len(frame_times), total))
