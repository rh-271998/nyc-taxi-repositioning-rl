import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window):
    """Simple trailing moving average; returns an array of length len(x)-window+1."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or window > len(x):
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_return_curve(returns,
                      savePath='../result/learning_curve.png',
                      window=100,
                      title='Training return vs episode',
                      baseline=None,
                      baselineLabel='random baseline'):
    """
    Plot per-episode training return with a smoothed trend line.

    returns       : 1D sequence of per-episode total returns
    savePath      : where to write the PNG
    window        : moving-average window (episodes) for the trend line
    baseline      : optional horizontal reference (e.g. random-policy mean)
    """
    returns = np.asarray(returns, dtype=float)
    episodes = np.arange(len(returns))

    fig, ax = plt.subplots(figsize=(9, 5))

    # raw, noisy per-episode returns (faint)
    ax.plot(episodes, returns, color='#9ecae1', alpha=0.35, linewidth=0.8,
            label='per-episode return')

    # smoothed trend (bold) — this is what shows the climb + plateau
    if window > 1 and window <= len(returns):
        smoothed = moving_average(returns, window)
        ax.plot(episodes[window - 1:], smoothed, color='#08519c', linewidth=2.0,
                label=f'{window}-episode moving avg')

    # optional baseline reference line
    if baseline is not None:
        ax.axhline(baseline, color='#e6550d', linestyle='--', linewidth=1.5,
                   label=baselineLabel)

    ax.set_xlabel('episode')
    ax.set_ylabel('total return ($ / day)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savePath, dpi=150)
    print(f'saved learning curve to {savePath}')
    return fig
