import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.utils import (
    get_general_path,
    join_paths,
)


FIGURES_PATH = 'reports/figures'


# Metrics
def evaluate_metric(y_true, y_pred, metric, metric_name=''):
    score = metric(y_true, y_pred)
    print(f'The {metric_name} score is: {score}')
    return score


# Plots
def plot_histogram(y_true, y_pred, fig_name):
    general_path = get_general_path()
    fig_path = join_paths(general_path, FIGURES_PATH, fig_name)
    steps = 50
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        y_pred[y_true.similar == 0],
        bins=np.linspace(0, 1, steps),
        stat="probability",
        alpha=0.5,
        label='Similarity=0'
    )
    sns.histplot(
        y_pred[y_true.similar == 1],
        bins=np.linspace(0, 1, steps),
        stat="probability",
        alpha=0.5,
        label='Similarity=1'
    )
    plt.legend()
    plt.title("Histogram normalized to the probability")
    plt.xlabel("Model Score")
    plt.ylabel("Count normalized to class probability")
    plt.savefig(fig_path)


def plot_proportion_of_similarity_in_bins(y_true, y_pred, fig_name):
    general_path = get_general_path()
    fig_path = join_paths(general_path, FIGURES_PATH, fig_name)
    steps = 25
    results = pd.DataFrame()
    results['pred'] = y_pred
    results['similar'] = y_true

    results['bins'] = pd.cut(results.pred, bins=np.linspace(0, 1, steps))
    fig, ax = plt.subplots(figsize=(10, 5))
    results.groupby("bins").similar.mean().plot(
        ax=ax,
        label='similar proportion'
    )
    ax.set_title("Proportion of similarity in bins")
    ax.set_ylim(0, 1)
    ax.grid()
    ax2 = plt.twinx(ax)
    results.groupby('bins').similar.count().plot(
        ax=ax2, kind='bar', color='orange', alpha=0.5, label='count'
    )
    ax.set_xticks(np.linspace(0, steps, steps - 1))
    ax2.set_xlim(-1, steps - 1)
    ax2.set_ylabel('Count')
    ax.set_ylabel('Similar proportion of cases')
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1.0))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.05, 0.9))
    plt.savefig(fig_path)
