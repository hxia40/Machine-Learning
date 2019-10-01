import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mlrose


def nCityTSP(city_num=10):
    '''Problem 1: n-city TSP: over a map of given size,
    generate N cities for a salesman to travel through each city and find the shortest route'''
    # Create list of city coordinates
    np.random.seed(0)
    coords_list = []
    for i in range(city_num):
        coords_list.append((np.random.random_sample(), np.random.random_sample()))

    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords=coords_list)

    # Define optimization problem object
    problem = mlrose.TSPOpt(length=city_num, fitness_fn=fitness_coords, maximize=False)

    "========simulated annealing========"
    best_fitness_SA_list = []
    best_fitness_SA_att_list = []
    alter_list = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 50):
        best_state_SA, best_fitness_SA = mlrose.simulated_annealing(problem,
                                                                    schedule=mlrose.ExpDecay(),
                                                                    max_attempts=100,
                                                                    max_iters=i,
                                                                    random_state=1)
        best_state_SA_att, best_fitness_SA_att = mlrose.simulated_annealing(problem,
                                                                            schedule=mlrose.ExpDecay(),
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            random_state=1)
        best_fitness_SA_list.append(best_fitness_SA)
        best_fitness_SA_att_list.append(best_fitness_SA_att)
        alter_list.append(i)
    # plotting
    plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)
    # plt.fill_between(alter, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(alter, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(alter_list, best_fitness_SA_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('SA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list, best_fitness_SA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('SA_max_attempt.png')
    plt.gcf().clear()
    "========MIMIC========"
    # best_state_MI, best_fitness_MI = mlrose.algorithms.mimic(problem,
    #                                                    pop_size=200,
    #                                                    keep_pct=0.2,
    #                                                    max_attempts=100,
    #                                                    max_iters=np.inf,
    #                                                    curve=False,
    #                                                    random_state = 1)
    # print(best_state_MI, best_fitness_MI)
    "========GA========"
    # best_state, best_fitness = mlrose.genetic_alg(problem,
    #                                               pop_size=200,
    #                                               mutation_prob=0.1,
    #                                               max_attempts=1000,
    #                                               max_iters=np.inf,
    #                                               curve=False,
    #                                               random_state=0)
    "========random hill climbing========"
    # best_state, best_fitness = mlrose.random_hill_climb(problem,
    #                                                     max_attempts=1000,
    #                                                     max_iters=np.inf,
    #                                                     restarts=0,
    #                                                     init_state=None,
    #                                                     curve=False,
    #                                                     random_state=1)


if __name__=="__main__":
    nCityTSP(city_num=10)





