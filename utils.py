"""A module with util functions."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.fft import fft

rcParams.update({
    'figure.subplot.left': 0.1,
    'toolbar': 'None'
})

class Plotter(object):
    _PAUSE_TIME = 0.03

    def __init__(self) -> None:
        fig, self._axes = plt.subplots(3, 1, figsize=(10, 10))
        fig.canvas.manager.set_window_title('Performance Innovative Task')

        def event_callback(event):
            if event.key == 'escape':
                sys.exit(0)

        fig.canvas.mpl_connect('key_press_event', event_callback)

        plt.show(block=False)

    def plot(self, result, time_domain, frequency_domain) -> None:
        self._axes[0].cla()
        self._axes[1].cla()
        self._axes[2].cla()

        self._plot_classification(result, self._axes[0])
        self._plot_time_domain(time_domain, self._axes[1])
        self._plot_frequency_domain(frequency_domain, self._axes[2])

        plt.tight_layout()
        plt.draw()

        plt.pause(self._PAUSE_TIME)

    def _plot_classification(self, result, ax) -> None:
        ax.set_title('Audio Classification')
        ax.set_ylim((0, 1))

        classification = result.classifications[0]
        label_list = [category.category_name for category in classification.categories]
        score_list = [category.score for category in classification.categories]

        ax.bar(label_list, score_list)
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('Confidence Score')

    def _plot_time_domain(self, time_domain, ax) -> None:
        ax.set_title('Time Domain Representation')
        ax.plot(time_domain)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    def _plot_frequency_domain(self, frequency_domain, ax) -> None:
        ax.set_title('Frequency Domain Representation')
        ax.plot(frequency_domain)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')