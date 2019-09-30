import mlrose
import numpy as np


fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)

"========Queens - simulated annealing========"
# Define decay schedule
schedule = mlrose.ExpDecay()
# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# Solve problem using simulated annealing
# best_state, best_fitness = mlrose.simulated_annealing(problem,
#                                                       schedule = schedule,
#                                                       max_attempts = 100,
#                                                       max_iters =1000,
#                                                       random_state = 1)
"========Queens - MIMIC========"
# best_state, best_fitness = mlrose.algorithms.mimic(problem,
#                                                    pop_size=200,
#                                                    keep_pct=0.2,
#                                                    max_attempts=100,
#                                                    max_iters=np.inf,
#                                                    curve=False,
#                                                    random_state = 1)
"========Queens - GA========"
# best_state, best_fitness = mlrose.genetic_alg(problem,
#                                               pop_size=200,
#                                               mutation_prob=0.1,
#                                               max_attempts=1000,
#                                               max_iters=np.inf,
#                                               curve=False,
#                                               random_state=0)
"========Queens - random hill climbing========"
best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                    max_attempts=1000,
                                                    max_iters=np.inf,
                                                    restarts=0,
                                                    init_state=None,
                                                    curve=False,
                                                    random_state=1)



print(best_state)

print(best_fitness)
