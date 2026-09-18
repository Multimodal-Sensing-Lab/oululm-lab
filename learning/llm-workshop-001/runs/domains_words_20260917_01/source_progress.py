"""A small plot that is saved during training; optionally show a live window."""
from pathlib import Path


class LossPlot:
    def __init__(self, path, title='Next-token training loss', live=False):
        import matplotlib
        if not live:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt, self.path, self.title, self.live = plt, Path(path), title, live
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        if live:
            plt.ion()
            plt.show(block=False)

    def update(self, history, keys=('train', 'validation')):
        self.ax.clear()
        for key in keys:
            points = [p for p in history if p.get(key) is not None]
            if points:
                self.ax.plot([p['step'] for p in points], [p[key] for p in points], label=key.replace('_', ' '))
        self.ax.set(xlabel='Optimizer updates', ylabel='Cross-entropy (nats per target token)', title=self.title)
        self.ax.grid(alpha=.2)
        self.ax.legend()
        self.fig.tight_layout()
        # Replace atomically so a browser never reads a half-written PNG.
        temporary = self.path.with_name(self.path.stem + '.tmp.png')
        self.fig.savefig(temporary, dpi=140)
        temporary.replace(self.path)
        if self.live:
            self.fig.canvas.draw_idle()
            self.plt.pause(.01)

    def close(self):
        self.fig.savefig(self.path.with_suffix('.svg'))
        self.plt.close(self.fig)
