import gym
import hiive.mdptoolbox.mdp as mdp
from hiive.mdptoolbox.example import forest
import numpy as np


def get_results(algos: dict):
    run_stats = [algos[algo].run() for algo in algos]

    ret = {}

    algos['ql'].iter = algos['ql'].max_iter
    ret['iters'] = {algo: np.array([algos[algo].iter]) for algo in algos}
    ret['policy'] = {algo : algos[algo].policy for algo in algos}
    ret['value'] = {algo : algos[algo].V for algo in algos}

    return ret


small_env = gym.make('FrozenLake8x8-v0')
small_t = np.zeros((small_env.nA, small_env.nS, small_env.nS))
small_r = np.copy(small_t)
for state in range(small_env.nS):
    for action in range(small_env.nA):
        for prob, next_state, reward, _ in small_env.P[state][action]:
            small_t[action, state, next_state] += prob
            small_r[action, state, next_state] += reward

big_t, big_r = forest(1000)

if __name__ == '__main__':
    small_algos = {
        'vi' : mdp.ValueIteration(small_t, small_r, 0.9),
        'pi' : mdp.PolicyIteration(small_t, small_r, 0.9),
        'ql' : mdp.QLearning(small_t, small_r, 0.9)
    }

    big_algos = {
        'vi' : mdp.ValueIteration(big_t, big_r, 0.9),
        'pi' : mdp.PolicyIteration(big_t, big_r, 0.9),
        'ql' : mdp.QLearning(big_t, big_r, 0.9)
    }

    big_results = get_results(big_algos)
    small_results = get_results(small_algos)

    for out_type in big_results:
        for algo in big_results[out_type]:
            np.savetxt('_'.join(['rl_images\large', algo, out_type]) + '.csv', big_results[out_type][algo], delimiter=',')
            np.savetxt('_'.join(['rl_images\small', algo, out_type]) + '.csv', small_results[out_type][algo], delimiter=',')
            if out_type == 'value':
                print(algo)
                for i in range(8):
                    print(' '.join(format(x, '.5f') for x in small_results[out_type][algo][i * 8: (i + 1) * 8]))
                print()

    small_policies = {
        'pi' : [['' for j in range(8)] for i in range(8)],
        'vi' : [['' for j in range(8)] for i in range(8)],
        'ql' : [['' for j in range(8)] for i in range(8)]
    }

    for algo in small_results['policy']:
        for i in range(8):
            for j in range(8):
                val = small_results['policy'][algo][i * 8 + j]
                if val == 0:
                    small_policies[algo][i][j] = 'L'
                elif val == 1:
                    small_policies[algo][i][j] = 'D'
                elif val == 2:
                    small_policies[algo][i][j] = 'R'
                elif val == 3:
                    small_policies[algo][i][j] = 'U'
        print(algo)
        for row in small_policies[algo]:
            print(''.join(row))
        print()