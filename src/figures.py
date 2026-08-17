"""
Portfolio figures for the NYC taxi repositioning project.
Numbers are the final eval results (1000 paired seeds). Edit here if re-run.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUT = '../result/plots'
os.makedirs(OUT, exist_ok=True)

# colorblind-safe palette (Okabe-Ito)
C_RANDOM  = '#999999'
C_TABULAR = '#0072B2'
C_DQN     = '#E69F00'
C_DDQN    = '#009E73'
C_PRED    = '#CC79A7'

# ---- final results (1000 paired seeds) ----
RET = {  # policy: (mean, se)
    'Random':      (178.6, 2.3),
    'Tabular Q':   (922.4, 8.8),
    'DQN':         (903.4, 9.2),
    'Double DQN':  (928.6, 9.1),
}
DIAG = {  # agent: (predicted, realized, bias, biasSE)
    'DQN':        (1001.6, 903.4, 98.1, 4.8),
    'Double DQN': (944.1,  928.6, 15.6, 4.0),
}


def fig_returns():
    names = list(RET.keys())
    means = [RET[n][0] for n in names]
    ses   = [RET[n][1] for n in names]
    colors = [C_RANDOM, C_TABULAR, C_DQN, C_DDQN]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, means, yerr=ses, capsize=5, color=colors,
                  edgecolor='white', linewidth=1.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 14, f'${m:.0f}',
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Mean daily earnings ($ / shift)')
    ax.set_title('Learned policies earn ~5× the random baseline\n(1000 paired seeds, greedy, ±SE)')
    ax.set_ylim(0, 1050)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_returns.png', dpi=150)
    print(f'saved {OUT}/fig_returns.png')


def fig_overestimation():
    agents = list(DIAG.keys())
    x = np.arange(len(agents))
    w = 0.36

    pred = [DIAG[a][0] for a in agents]
    real = [DIAG[a][1] for a in agents]
    bias = [DIAG[a][2] for a in agents]
    biasSE = [DIAG[a][3] for a in agents]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - w/2, pred, w, label='Predicted value  Q(s₀)',
                color=C_PRED, edgecolor='white', linewidth=1.5)
    b2 = ax.bar(x + w/2, real, w, label='Realized return',
                color=C_TABULAR, edgecolor='white', linewidth=1.5)

    # value labels on the bars
    for bars, vals in [(b1, pred), (b2, real)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+10, f'${v:.0f}',
                    ha='center', va='bottom', fontsize=9, color='#444')

    # bias called out well above each group (no crowding)
    for i in range(len(agents)):
        ax.text(x[i], 1085, f'overestimates\n+${bias[i]:.0f} ± {biasSE[i]:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11,
                color='#B00020' if bias[i] > 50 else '#00695C')

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    ax.set_ylabel('Whole-shift value ($)')
    ax.set_title('Double DQN cuts value overestimation by ~84%\n'
                 'predicted vs. realized whole-day earnings (1000 seeds)',
                 pad=38)
    ax.set_ylim(0, 1240)
    ax.legend(loc='center right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_overestimation.png', dpi=150)
    print(f'saved {OUT}/fig_overestimation.png')


if __name__ == '__main__':
    fig_returns()
    fig_overestimation()