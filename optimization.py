# https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
import matplotlib.pyplot as plt
import mlrose_hiive as rose
import numpy as np

def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

fitness_funcs = [
    {
        'name' : 'Queens',
        'func' : rose.CustomFitness(queens_max),
        'len'  : 8,
        'val'  : 8
    },
    {
        'name' : 'Continuous Peaks',
        'func' : rose.ContinuousPeaks(),
        'len'  : 100,
        'val'  : 2
    },
    {
        'name' : 'Flip Flop',
        'func' : rose.FlipFlop(),
        'len'  : 1000,
        'val'  : 2
    },
]

problems = [rose.DiscreteOpt(length=x['len'], fitness_fn = x['func'], max_val=x['val']) for x in fitness_funcs]

algos = [
    {
        'name' : 'RHC',
        'algo' : rose.random_hill_climb,
        'kwargs' : {
            'restarts' : 10,
        }
    },
    {
        'name' : 'SA',
        'algo' : rose.simulated_annealing,
        'kwargs' : { }
    },
    {
        'name' : 'GA',
        'algo' : rose.genetic_alg,
        'kwargs' : { }
    },
    {
        'name' : 'MIMIC',
        'algo' : rose.mimic,
        'kwargs' : { }
    }
]


def get_filename(algo_name, fitness_name, suffix):
    tail = [suffix] if suffix else []
    return '_'.join(['optimized', algo_name, fitness_name] + tail) + '.txt'

def get_discrete_data(suffix=''):
    results = [[x['algo'](problem, curve=True, random_state=12345, **x['kwargs']) for problem in problems] for x in algos]
    for i, algo in enumerate(results):
        for j, problem in enumerate(algo):
            with open(get_filename(algos[i]['name'], fitness_funcs[j]['name'], suffix), 'w') as f:
                np.savetxt(f, problem[0], delimiter=',')
                f.write(str(problem[1]) + '\n')
                np.savetxt(f, problem[2], delimiter=',')

def get_discrete_graphs(suffix=''):
    for algo in algos:
        for func in fitness_funcs:
            with open(get_filename(algo['name'], func['name'], suffix), 'r') as f:
                best_state = np.loadtxt(f, delimiter=',', max_rows=func['len'])
                best_fitness = float(f.readline())
                fitness_curve = np.loadtxt(f, delimiter=',')

                plt.title(' '.join([algo['name'], 'Performance on', func['name']]))
                fig, ax1 = plt.subplots()

                color = 'red'
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Fitness', color=color)
                ax1.plot(fitness_curve[:,1], fitness_curve[:,0], color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                # color = 'blue'

                # ax2 = ax1.twinx()
                # ax2.set_ylabel('# of Function Evaluations', color=color)
                # ax2.plot(fitness_curve[:,1], color=color)
                # ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()
                plt.savefig(get_filename(algo['name'], func['name'], suffix)[:-3] + 'png')


if __name__ == '__main__':
    get_discrete_graphs(suffix='long')