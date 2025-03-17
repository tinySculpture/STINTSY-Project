import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def plot_distribution(data, column, ax, is_categorical):
    if is_categorical:
        sns.countplot(data[column], ax=ax)
    else:
        sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(column)

def plot_all_distributions(data):
    print("Plotting all distributions...")
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    numerical_columns = data.select_dtypes(exclude=["object", "category"]).columns

    columns = 3
    rows = (len(data.columns) + columns - 1)
    fig, axes = plt.subplots(rows, columns, figsize=(15, rows  * 5))
    axes = axes.flatten()

    i = 0
    for column in categorical_columns:
        plot_distribution(data, column, axes[i], is_categorical=True)
        i += 1

    for column in numerical_columns:
        plot_distribution(data, column, axes[i], is_categorical=False)
        i += 1

    for j in range(i, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def check_normality(data, column):
    print("Checking normality...")

    # Kolmorogov-Smirnov Test
    stat, p = stats.kstest(data[column].dropna(), 'norm')
    print(f'Kolmogorov-Smirnov Test for {column}: Statistics={stat}, p={p}')

    #Q-Q plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f'{column} Histogram')

    plt.subplot(1, 2, 2)
    stats.probplot(data[column].dropna(), plot=plt, dist="norm")
    plt.title(f'{column} Q-Q Plot')

    plt.tight_layout()
    plt.show()

def plot_all_normality_checks(data):
    for column in data.select_dtypes(include=[np.number]).columns:
        check_normality(data, column)