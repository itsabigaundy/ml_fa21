from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils


def runTree(
    df_dict : Dict[ str, Tuple[ pd.DataFrame, pd.DataFrame ], ],
    train_range : np.array,
    seed_num : int,
) -> None:
    alpha_range = np.linspace(0, 1, 11)

    for name, (X, y) in df_dict.items():
        test_mean = []
        test_max = []
        for alpha in alpha_range:
            dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
            train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(dtree, X, y, train_sizes=train_range, shuffle=True, random_state=seed_num, return_times=True)

            mean_train = np.mean(train_scores, axis=-1)
            mean_test = np.mean(test_scores, axis=-1)

            test_mean.append(np.mean(test_scores))
            test_max.append(np.max(mean_test))

            plt.plot(train_sizes, mean_train, '-o', label='Mean Training score')
            plt.plot(train_sizes, mean_test, '-o', label='Mean Testing score')
            plt.legend()
            plt.xlabel('Training samples')
            plt.ylabel('Score')

            plt.savefig('{}_score_vs_train_samples_alpha_{:.2f}.png'.format(name, alpha))
            plt.clf()

        plt.plot(alpha_range, test_max, '-o', label='Max Testing Score')
        plt.plot(alpha_range, test_mean, '-o', label='Mean Testing score')
        plt.legend()
        plt.xlabel('MCCP Parameter ($\\alpha$)')
        plt.ylabel('Score')
        plt.savefig('{}_score_vs_alpha.png'.format(name))
        plt.clf()


def runKNN(
    df_dict : Dict[ str, Tuple[ pd.DataFrame, pd.DataFrame ], ],
    train_range : np.array,
    seed_num : int,
) -> None:
    k_range = list(range(1, 11))
    for name, (X, y) in df_dict.items():
        test_mean = []
        test_max = []
        for k in k_range:
            knn = KNeighborsClassifier(k)
            train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(knn, X, y, train_sizes=train_range, shuffle=True, random_state=seed_num, return_times=True)

            mean_train = np.mean(train_scores, axis=-1)
            mean_test = np.mean(test_scores, axis=-1)

            test_mean.append(np.mean(test_scores))
            test_max.append(np.max(mean_test))

            plt.plot(train_sizes, mean_train, '-o', label='Mean Training score')
            plt.plot(train_sizes, mean_test, '-o', label='Mean Testing score')
            plt.legend()
            plt.xlabel('Training samples')
            plt.ylabel('Score')

            plt.savefig('{}_score_vs_train_samples_k_{}.png'.format(name, k))
            plt.clf()

        plt.plot(k_range, test_max, '-o', label='Max Testing Score')
        plt.plot(k_range, test_mean, '-o', label='Mean Testing score')
        plt.legend()
        plt.xlabel('Number of Neighbors ($k$)')
        plt.ylabel('Score')
        plt.savefig('{}_score_vs_k.png'.format(name))
        plt.clf()


if __name__ == '__main__':
    credit_df = utils.getCreditRiskData()
    zoo_df = utils.getZooData()
    

    df_dict = {
        'risk' : ( credit_df.drop( [ 'label', ], axis=1 ), credit_df['label'] ),
        'zoo' : ( zoo_df.drop( [ 'class_type', ], axis=1 ), zoo_df['class_type'] ),
    }

    seed_num = 1738
    train_range = np.linspace(0.1, 1.0, 10)

    # runTree(df_dict, train_range, seed_num)
    runKNN(df_dict, train_range, seed_num)