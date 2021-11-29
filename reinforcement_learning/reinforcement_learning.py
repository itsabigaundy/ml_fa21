from typing import Callable, Iterable

import gym
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

from hiive.mdptoolbox.example import forest


small_env = gym.make('FrozenLake8x8-v1')
small_t = np.zeros((small_env.nA, small_env.nS, small_env.nS))
small_r = np.copy(small_t)
for state in range(small_env.nS):
    for action in range(small_env.nA):
        for prob, next_state, reward, _ in small_env.P[state][action]:
            small_t[action, state, next_state] += prob
            small_r[action, state, next_state] += reward


big_t, big_r = forest(1000)


def get_small_algos(gamma: float):
    return {
        'vi' : mdp.ValueIteration(small_t, small_r, gamma, epsilon=1e-5, max_iter=10000,),
        'pi' : mdp.PolicyIteration(small_t, small_r, gamma, eval_type=0, max_iter=10000,),
        'ql' : mdp.QLearning(small_t, small_r, gamma, n_iter=100000,)
    }


def get_big_algos(gamma: float):
    return {
        'vi' : mdp.ValueIteration(big_t, big_r, gamma, epsilon=1e-5, max_iter=10000,),
        'pi' : mdp.PolicyIteration(big_t, big_r, gamma, eval_type=0, max_iter=10000,),
        'ql' : mdp.QLearning(big_t, big_r, gamma, n_iter=100000,)
    }


def get_results(get_algos: Callable, gamma: float, num_runs: int):
    algos = get_algos(gamma)
    ret = {
        algo: {
            'iter': [],
            'time': [],
            'policy': [],
            'value': [],
        } for algo in algos
    }

    for _ in range(num_runs):
        algos = get_algos(gamma)
        for algo in algos:
            algos[algo].run()
            ret[algo]['iter'].append(algos[algo].iter if algo != 'ql' else algos[algo].max_iter)
            ret[algo]['time'].append(algos[algo].time)
            ret[algo]['policy'].append(np.array(algos[algo].policy))
            ret[algo]['value'].append(np.array(algos[algo].V))

    for algo in algos:
        ret[algo]['iter'] = np.mean(ret[algo]['iter'])
        ret[algo]['time'] = np.mean(ret[algo]['time'])
        ret[algo]['policy'] = sps.mode(ret[algo]['policy'])[0][0]
        ret[algo]['value'] = np.mean(ret[algo]['value'], axis=0)

    return ret


def getSingleGammaCase(gamma: float=0.9, num_runs: int=5):
    small_results = get_results(get_small_algos, gamma, num_runs,)
    big_results = get_results(get_big_algos, gamma, num_runs,)

    small_policies = {
        'pi' : [['' for j in range(8)] for i in range(8)],
        'vi' : [['' for j in range(8)] for i in range(8)],
        'ql' : [['' for j in range(8)] for i in range(8)]
    }

    for algo in small_results:
        for i in range(8):
            for j in range(8):
                val = small_results[algo]['policy'][i * 8 + j]
                if val == 0:
                    small_policies[algo][i][j] = 'L'
                elif val == 1:
                    small_policies[algo][i][j] = 'D'
                elif val == 2:
                    small_policies[algo][i][j] = 'R'
                elif val == 3:
                    small_policies[algo][i][j] = 'U'
        print(algo, small_results[algo]['time'], small_results[algo]['iter'], small_results[algo]['value'][0])
        for row in small_policies[algo]:
            print(''.join(row))
        print()

    for algo in big_results:
        print(algo, big_results[algo]['time'], big_results[algo]['iter'], big_results[algo]['value'][0])


def getGammaGraphs(gammas: np.array=np.linspace(0.1, 0.9, 9), num_runs: int=5):
    small_results = {gamma: get_results(get_small_algos, gamma, num_runs) for gamma in gammas}
    big_results = {gamma: get_results(get_big_algos, gamma, num_runs) for gamma in gammas}

    def plot_helper(algos: Iterable, results: dict, field: str, ylabel: str, title_field: str,):
        if field != 'value':
            for algo in algos:
                plt.plot(gammas, [results[gamma][algo][field] for gamma in gammas], label=algo)
        else:
            for algo in algos:
                plt.plot(gammas, [results[gamma][algo]['value'][0] for gamma in gammas], label=algo)
        plt.xlabel('$\\gamma$')
        plt.ylabel(ylabel)
        plt.title('Average {} over {} Runs vs $\\gamma$'.format(title_field, num_runs,))
        plt.legend()
        plt.savefig('{}_{}.png'.format('small' if results is small_results else 'big', field,))
        plt.clf()
            
    plot_helper(('vi', 'pi',), small_results, 'iter', 'Iterations', 'Iterations',)
    plot_helper(('vi', 'pi',), small_results, 'time', 'Time (seconds)', 'Elapsed Time',)
    plot_helper(('vi', 'pi', 'ql',), small_results, 'value', 'Value', 'Value of Starting State',)

    plot_helper(('vi', 'pi',), big_results, 'iter', 'Iterations', 'Iterations',)
    plot_helper(('vi', 'pi',), big_results, 'time', 'Time (seconds)', 'Elapsed Time',)
    plot_helper(('vi', 'pi', 'ql',), big_results, 'value', 'Value', 'Value of Starting State',)


if __name__ == '__main__':
    getSingleGammaCase()
    getGammaGraphs()