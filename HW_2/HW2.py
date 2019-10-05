import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mlrose
import time
"""
4:20
Q:  Can we use a stochastic fitness function instead of a deterministic one for the bitstring evaluation? I mean that 
a fitness function would return different results for the same state. 
 
A: So those optimizations proboelms are all well-known. basically u want to see how they work.
there are two of them are instance-based optimization, 
RHC and SA, and there are to distribution-based, GA and MIMIC. u want to see how those specific optimization problem
 works where they have specific structures: number of local maximal/minimal, global maximal/minimal, or is it 
 NP-hard problems, u know, each of them problems has a specific type of characteristics, and some of these random  
optimizations are gonna better for type of problem. So we really want you to talk about that.
   
6:35   
Q1.  When we compare the number of iterations/evaluations (and run times) of different algorithms wrt to a problem input
 sizes,  I believe we need to take into the fitness scores.  It doesn't seem to make sense to say algorithm A has fewer 
 iterations than algorithm B if the fitness score of A is much lower than B. 
Any thoughts?

One thing that is different on this assignment is, not only we care about to get higher fitness score, but we also care
about the time, like how long and how many iterations .Cause if you keep to running it, at some point you can get the 
 optimal answer, but u dont want to do it forever. So the point is that if u want to compare two optimiazation methods,u 
 need to compared two aspects. one is finess score, the other is time/iterations/evulatations. If one method is 
 to get answer very fast,  but score is very low, thats not sth u will want, right? but still u would want to talk 
 about both aspects . just keep in mind to make comparation on both of the aspects when u make the cmaration.
 
Q2.  On a related note, when we vary the input size of a problem, should we keep the parameters (eg population size) of 
the algorithm constant at different input problem sizes?

A: Yeah, i would pro do that. u can start with like w/e setting of the problem, find to make these methods better, just 
stick w/ them. 
Cause we dont really want u to change everything. For example, when u are modifying input size e.g. its getting bigger,
 u would want to keep h-parameter constant, u may see some part interesting behaviors here, how the problems getting 
 harder, and our algorithms responding in different ways. 

Q3.  Since this is a random optimization assignment, it seems that we should not set the random seed for a run but 
instead should do multiple runs and "average" the output results.  What are your recommendations?

A: yeah thats true, u can do some runs and average the results. so do it for more than just once. doesnt need to be 
huge number, just a resonable number

18:20
Q:Just curious, is there a problem that Genetic Algorithm is better than MIMIC?
Edit: yes, please ignore. I think I know the answer (was previously stuck in local optimal, lol)

A: again, there are two aspects, one is to get the optimal answer, and the second one is ..not only about how many 
evaluation, as MIMIC usually takes fewer evaluations, but also  about the time it takes.. so yeah u just need to think 
aobut one of those problems , depend on what GA does, who actually uses distribution different than MIMIC does, and 
there will be diffrence as what charles said, theres no free lunch. 

22:30
Q:Also, I want to make sure the first randomized optimization: randomized hill climbing is the randomized restart hill 
climbing like mentioned in the lecture.
So the randomness come from the (re)starting point but not the random neighbor within each hill climbing.
 
Asking this because in Python mlrose, there is hillclimb (where we can set # of restart) and random_hill_climb use 
random_neighbor instead of best_neighbor. Back to my question, does this assignment wants us to use the randomized 
restart hill climbing as in lecture (not randomized neighbor)?

A: actually we meant the non-restart version. u know, SA has a way to get out of.. local optima, but RHC w/o restart
 will probably perform poorly. but that's fine , we want u to actually see how poorly it can perform. and if u want to 
 add the restart, u ae welcome to, but if u implement the non-restart version ONLY, u are fine - u wont be penalized. u
 can do both though.
 
24:30
Q: I am really confused about what graphs and charts would be helpful to include for Assignment 2: 
Randomized Optimization. For the optimization problems would a comparison across the 4 RS algos in terms of times and 
iterations would be helpful. Some examples or reading would really help focus on the correct things.I am also assuming 
that for ANN, we would focus on the same set of charts / diagrams as Assignment 1: supervised Learning. 
Is is also expected that we compare the back-propagation method with randomized search? 
Thanks

A: Ok, yeah, in terms of graphs and chart, if u think about the goal here, u need to set some parameters in the first
 section, that might need some chart. and then u need to compare time and accuracy, then u will need some charts. that's
 the main one. if u want to include some charts to support ur points, thats fine. And for NN, actually the whole point 
 of NN is that u comepare Back propagation with those random search , so we want u to thicnk to think about 
 how is Back propagation is diffetn from random search methods, in terms of being continuous /discrete, Back propagation 
 there is a loop while the other dont. just look at the result ant think whats happening and why its happening. 
 . thats the main goal. U will be using the same network structure as assignment one. u 
 dont need to include any type of learning curve or model compexlity curve, but theres still some parameter tunning for
 the optimizatio methods, and then u again are talking about the accuracy aspect and run time aspect-
 this is rather discussion, rather than chart. 
"""


def RHC(problem, prob_name):

    best_fitness_RHC_list = []
    best_fitness_RHC_att_list = []
    alter_list_RHC = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 20):
        print(prob_name, "RHC:", i)
        best_state_RHC, best_fitness_RHC = mlrose.random_hill_climb(problem,
                                                                    max_attempts=10,
                                                                    max_iters=i,
                                                                    restarts=0,
                                                                    init_state=None,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_RHC_att, best_fitness_RHC_att = mlrose.random_hill_climb(problem,
                                                                            max_attempts=i,
                                                                            max_iters=10,
                                                                            restarts=0,
                                                                            init_state=None,
                                                                            curve=False,
                                                                            random_state=1)
        best_fitness_RHC_list.append(best_fitness_RHC)
        best_fitness_RHC_att_list.append(best_fitness_RHC_att)
        alter_list_RHC.append(i)
    alter_list_RHC = np.array(alter_list_RHC)
    best_fitness_RHC_list = np.array(best_fitness_RHC_list)
    best_fitness_RHC_att_list = np.array(best_fitness_RHC_att_list)

    return np.array([[alter_list_RHC, best_fitness_RHC_list,"max_iters", "best_fitness"],
           [alter_list_RHC, best_fitness_RHC_att_list,"max_attempts", "best_fitness"]])
def SA(problem, prob_name):

    best_fitness_SA_list = []
    best_fitness_SA_att_list = []
    alter_list_SA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 20):
        print(prob_name, "SA:", i)
        best_state_SA, best_fitness_SA = mlrose.simulated_annealing(problem,
                                                                    schedule=mlrose.ExpDecay(),
                                                                    max_attempts=10,
                                                                    max_iters=i,
                                                                    random_state=1)
        best_state_SA_att, best_fitness_SA_att = mlrose.simulated_annealing(problem,
                                                                            schedule=mlrose.ExpDecay(),
                                                                            max_attempts=i,
                                                                            max_iters=10,
                                                                            random_state=1)
        best_fitness_SA_list.append(best_fitness_SA)
        best_fitness_SA_att_list.append(best_fitness_SA_att)
        alter_list_SA.append(i)
    alter_list_SA = np.array(alter_list_SA)
    best_fitness_SA_list = np.array(best_fitness_SA_list)
    best_fitness_SA_att_list = np.array(best_fitness_SA_att_list)

    return np.array([[alter_list_SA, best_fitness_SA_list, "max_iters", "best_fitness"],
           [alter_list_SA,best_fitness_SA_att_list, "max_attempts", "best_fitness"]])
def GA(problem, prob_name):

    best_fitness_GA_list = []
    best_fitness_GA_att_list = []
    best_fitness_GA_pop_list = []
    best_fitness_GA_mutpb_list = []
    alter_list_GA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 20):
        print(prob_name, "GA:", i)
        start_time = time.time()
        best_state_GA, best_fitness_GA = mlrose.genetic_alg(problem,
                                                            pop_size=200,
                                                            mutation_prob=0.1,
                                                            max_attempts=10,
                                                            max_iters=i,
                                                            curve=False,
                                                            random_state=0)
        best_state_GA_att, best_fitness_GA_att = mlrose.genetic_alg(problem,
                                                                    pop_size=200,
                                                                    mutation_prob=0.1,
                                                                    max_attempts=i,
                                                                    max_iters=10,
                                                                    curve=False,
                                                                    random_state=0)
        best_state_GA_pop, best_fitness_GA_pop = mlrose.genetic_alg(problem,
                                                                    pop_size=max(1, int(i / 5)),
                                                                    mutation_prob=0.1,
                                                                    max_attempts=10,
                                                                    max_iters=10,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_GA_mutpb, best_fitness_GA_mutpb = mlrose.genetic_alg(problem,
                                                                        pop_size=200,
                                                                        mutation_prob=float(i) / 1000.0,
                                                                        max_attempts=10,
                                                                        max_iters=10,
                                                                        curve=False,
                                                                        random_state=1)
        best_fitness_GA_list.append(best_fitness_GA)
        best_fitness_GA_att_list.append(best_fitness_GA_att)
        best_fitness_GA_pop_list.append(best_fitness_GA_pop)
        best_fitness_GA_mutpb_list.append(best_fitness_GA_mutpb)
        alter_list_GA.append(i)
        end_time = time.time()
        print(end_time - start_time)
    alter_list_GA = np.array(alter_list_GA)
    best_fitness_GA_list = np.array(best_fitness_GA_list)
    best_fitness_GA_att_list = np.array(best_fitness_GA_att_list)
    best_fitness_GA_pop_list = np.array(best_fitness_GA_pop_list)
    best_fitness_GA_mutpb_list = np.array(best_fitness_GA_mutpb_list)

    return np.array([[alter_list_GA, best_fitness_GA_list, "max_iters", "best_fitness"],
           [alter_list_GA, best_fitness_GA_att_list, "max_attempts", "best_fitness"],
           [alter_list_GA/5, best_fitness_GA_pop_list, "pop_size", "best_fitness"],
           [alter_list_GA/1000, best_fitness_GA_mutpb_list, "mutation_prob", "best_fitness"]])
def MI(problem, prob_name):
    best_fitness_MI_list = []
    best_fitness_MI_att_list = []
    best_fitness_MI_pop_list = []
    best_fitness_MI_pct_list = []
    alter_list_MI = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 20):
        print(prob_name, "MI:", i)
        start_time = time.time()
        try:
            best_state_MI, best_fitness_MI = mlrose.algorithms.mimic(problem,
                                                                     pop_size=200,
                                                                     keep_pct=0.2,
                                                                     max_attempts=10,
                                                                     max_iters=i,
                                                                     curve=False,
                                                                     random_state=1)
        except:
            print("MI_iter error on i = {}".format(i))
            best_state_MI = np.nan
            best_fitness_MI = np.nan
        try:
            best_state_MI_att, best_fitness_MI_att = mlrose.algorithms.mimic(problem,
                                                                             pop_size=200,
                                                                             keep_pct=0.2,
                                                                             max_attempts=i,
                                                                             max_iters=10,
                                                                             curve=False,
                                                                             random_state=1)
        except:
            print("MI_att error on i = {}".format(i))
            best_state_MI_att = np.nan
            best_fitness_MI_att = np.nan
        try:
            best_state_MI_pop, best_fitness_MI_pop = mlrose.algorithms.mimic(problem,
                                                                             pop_size=max(1, int(i / 5)),
                                                                             keep_pct=0.2,
                                                                             max_attempts=10,
                                                                             max_iters=10,
                                                                             curve=False,
                                                                             random_state=1)
        except:
            print("MI_pop error on i = {}".format(i))
            best_state_MI_pop = np.nan
            best_fitness_MI_pop = np.nan
        try:
            best_state_MI_pct, best_fitness_MI_pct = mlrose.algorithms.mimic(problem,
                                                                             pop_size=200,
                                                                             keep_pct=max(0.1, float(i-1) / 1000.0),
                                                                             # keep_pct = 0.333,
                                                                             max_attempts=10,
                                                                             max_iters=10,
                                                                             curve=False,
                                                                             random_state=1)
        except:
            print("MI_pct error on i = {}".format(i))
            best_state_MI_pct = np.nan
            best_fitness_MI_pct = np.nan
        print('checker:', i, max(0.1, float(i +3) / 1000.0))
        best_fitness_MI_list.append(best_fitness_MI)
        best_fitness_MI_att_list.append(best_fitness_MI_att)
        best_fitness_MI_pop_list.append(best_fitness_MI_pop)
        best_fitness_MI_pct_list.append(best_fitness_MI_pct)
        alter_list_MI.append(i)

        end_time = time.time()
        print(end_time - start_time)
    alter_list_MI = np.array(alter_list_MI)
    best_fitness_MI_list = np.array(best_fitness_MI_list)
    best_fitness_MI_att_list = np.array(best_fitness_MI_att_list)
    best_fitness_MI_pop_list = np.array(best_fitness_MI_pop_list)
    best_fitness_MI_pct_list = np.array(best_fitness_MI_pct_list)

    return np.array([[alter_list_MI, best_fitness_MI_list, "max_iters", "best_fitness"],
           [alter_list_MI, best_fitness_MI_att_list, "max_attempts", "best_fitness"],
           [alter_list_MI/5, best_fitness_MI_pop_list, "pop_size", "best_fitness"],
           [alter_list_MI/1000, best_fitness_MI_pct_list, "keep_pct", "best_fitness"]])

def plotting(prob_name, algo_name, valuess, size_plot):
    size_plot = size_plot
    if (prob_name == "nCityTSP") or (prob_name == "MaxKColor"):
        print("sdfsdf")
        for value in valuess:
            if size_plot == True:
                if value[3] == "clock time":
                    writer = open('size_time/{}_{}_{}_{}.txt'.format(value[3], prob_name, algo_name, value[2]), 'w')
                else:
                    writer = open('size_fit/{}_{}_{}_{}.txt'.format(value[3], prob_name, algo_name, value[2]), 'w')
            else:
                writer = open('data/{}_{}_{}.txt'.format(prob_name, algo_name, value[2]), 'w')
            writer.write('{}_{}_{}'.format(prob_name, algo_name, value[2]))
            writer.write("\n\n")
            writer.write(str(value[0]))
            writer.write("\n\n")
            writer.write(str(1/value[1]))

            plt.plot(value[0], 1/value[1], color="r",)
            x_title = value[2]
            y_title = value[3]
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            if size_plot == True:
                if value[3] == "clock time":
                    plt.savefig('size_time/{}_{}_{}_{}.png'.format(value[3], prob_name, algo_name, value[2]))
                else:
                    plt.savefig('size_fit/{}_{}_{}_{}.png'.format(value[3], prob_name, algo_name, value[2]))
            else:
                plt.savefig('fig/{}_{}_{}.png'.format(prob_name, algo_name, value[2]))
            plt.gcf().clear()

    else:

        for value in valuess:
            if size_plot == True:
                if value[3] == "clock time":
                    writer = open('size_time/{}_{}_{}_{}.txt'.format(value[3], prob_name, algo_name, value[2]), 'w')
                else:
                    writer = open('size_fit/{}_{}_{}_{}.txt'.format(value[3], prob_name, algo_name, value[2]), 'w')
            else:
                writer = open('data/{}_{}_{}.txt'.format(prob_name, algo_name, value[2]), 'w')

            writer.write('{}_{}_{}'.format(prob_name, algo_name, value[2]))
            writer.write("\n\n")
            writer.write(str(value[0]))
            writer.write("\n\n")
            writer.write(str(value[1]))

            plt.plot(value[0], value[1], color="r",)
            x_title = value[2]
            y_title = value[3]
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            if size_plot == True:
                if value[3] == "clock time":
                    plt.savefig('size_time/{}_{}_{}_{}.png'.format(value[3], prob_name, algo_name, value[2]))
                else:
                    plt.savefig('size_fit/{}_{}_{}_{}.png'.format(value[3], prob_name, algo_name, value[2]))
            else:
                plt.savefig('fig/{}_{}_{}.png'.format(prob_name, algo_name, value[2]))
            plt.gcf().clear()

def generate_prob_MKC(random_seed=0, nodes=8, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        edges_list = []
        for i in range(nodes):
            for j in range(i + 1, nodes):
                if np.random.random() > 0.6:
                    edges_list.append((i, j))

        # Initialize fitness function object using edges_list
        fitness = mlrose.MaxKColor(edges_list)

        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=nodes, fitness_fn=fitness, maximize=False)
        problem_list.append(problem)
    return problem_list
def generate_prob_TSP(random_seed=0, citys=6, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        city_num = citys
        coords_list = []
        for i in range(city_num):
            coords_list.append((np.random.random_sample(), np.random.random_sample()))
        # rev_dist_list = []
        # for i in range(len(coords_list)):
        #     for j in range(i+1, len(coords_list)):
        #         rev_dist_list.append((i,j,
        #                 (2 ** 0.5)-(((coords_list[i][0]-coords_list[j][0]) ** 2 + (coords_list[i][1]-coords_list[j][1]) ** 2) ** 0.5)))
        # rev_dist_list = np.array(rev_dist_list)

        # Initialize fitness function object using coords_list
        fitness_cords = mlrose.TravellingSales(coords=coords_list)
        # fitness_dists = mlrose.TravellingSales(distances = rev_dist_list)

        # Define optimization problem object
        problem = mlrose.TSPOpt(length=city_num, fitness_fn=fitness_cords, maximize=False)
        problem_list.append(problem)
    return problem_list
def generate_prob_4peaks(random_seed=0, length = 10, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        # t_pct = np.random.random()
        fitness = mlrose.FourPeaks(t_pct=0.1)
        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        problem_list.append(problem)
    return problem_list
def generate_prob_Cpeaks(random_seed=0, length = 10, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        # length = int(np.random.randint(low=10, high=50, size=1))
        fitness = mlrose.ContinuousPeaks()
        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        problem_list.append(problem)
    return problem_list
def generate_prob_OneMax(random_seed=0, length = 10, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        # length = int(np.random.randint(low=10, high=50, size=1))
        fitness = mlrose.OneMax()
        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        problem_list.append(problem)
    return problem_list
def generate_prob_FlipFlop(random_seed=0, length = 10, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        # length = int(np.random.randint(low=10, high=50, size=1))
        fitness = mlrose.FlipFlop()
        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        problem_list.append(problem)
    return problem_list
def generate_prob_KnapSack(random_seed=0, length = 10, sample_size=5):
    np.random.seed(random_seed)
    problem_list = []
    for n in range(sample_size):
        weights = list(np.random.randint(low=10, high=50, size=length))
        values = list(np.random.randint(low=1, high=10, size=length))
        max_weight_pct = 0.6
        fitness = mlrose.Knapsack(weights, values, max_weight_pct)
        # Define optimization problem object
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
        problem_list.append(problem)
    return problem_list

def prob_to_curves(prob_name, problem_list):
    # Create list of city coordinates
    result_RHC_list = []
    result_SA_list = []
    result_GA_list = []
    result_MI_list = []

    for problem in problem_list:
        result_RHC = RHC(problem, prob_name)
        result_RHC_num = result_RHC[0:, :2]
        result_RHC_list.append(result_RHC_num)

        result_SA = SA(problem, prob_name)
        result_SA_num = result_SA[0:, :2]
        result_SA_list.append(result_SA_num)

        result_GA = GA(problem, prob_name)
        result_GA_num = result_GA[0:, :2]
        result_GA_list.append(result_GA_num)

        result_MI = MI(problem, prob_name)
        result_MI_num = result_MI[0:, :2]
        result_MI_list.append(result_MI_num)

    result_RHC_list = pd.DataFrame(result_RHC_list)
    result_RHC_list = result_RHC_list.fillna(method='ffill')
    result_RHC_mean = np.nanmean(result_RHC_list, axis=0)
    result_RHC_title = result_RHC[0:, 2:]
    results_RHC_complete = np.concatenate((result_RHC_mean[0], result_RHC_title), axis=1)
    plotting(prob_name, "RHC", results_RHC_complete)

    result_SA_list = pd.DataFrame(result_SA_list)
    result_SA_list = result_SA_list.fillna(method='ffill')
    result_SA_mean = np.nanmean(result_SA_list, axis=0)
    result_SA_title = result_SA[0:, 2:]
    results_SA_complete = np.concatenate((result_SA_mean[0], result_SA_title), axis=1)
    plotting(prob_name, "SA", results_SA_complete)

    result_GA_list = pd.DataFrame(result_GA_list)
    result_GA_list = result_GA_list.fillna(method='ffill')
    result_GA_mean = np.nanmean(result_GA_list, axis=0)
    result_GA_title = result_GA[0:, 2:]
    results_GA_complete = np.concatenate((result_GA_mean[0], result_GA_title), axis=1)
    plotting(prob_name, "GA", results_GA_complete)
    #
    result_MI_list = pd.DataFrame(result_MI_list)
    result_MI_list = result_MI_list.fillna(method='ffill')
    result_MI_mean = np.nanmean(result_MI_list, axis=0)
    result_MI_title = result_MI[0:, 2:]
    results_MI_complete = np.concatenate((result_MI_mean[0], result_MI_title), axis=1)
    plotting(prob_name, "MI", results_MI_complete)

def size_test_OneMax_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 1000, 100):
        problem_list = generate_prob_OneMax(random_seed=0, length = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_OneMax_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 1000, 100):
        problem_list = generate_prob_OneMax(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=10,
                                                                  max_iters=20,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_OneMax_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 1000, 100):
        problem_list = generate_prob_OneMax(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=100,
                                                          mutation_prob=0.5,
                                                          max_attempts=50,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_OneMax_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 1000, 100):
        print("onemax_mi:", i)
        problem_list = generate_prob_OneMax(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                               pop_size=50,
                                                               keep_pct=0.1,
                                                               max_attempts=10,
                                                               max_iters=10,
                                                               curve=False,
                                                               random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_4peaks_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_4peaks(random_seed=0, length = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_4peaks_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_4peaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=50,
                                                                  max_iters=50,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_4peaks_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_4peaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=50,
                                                          mutation_prob=0.5,
                                                          max_attempts=10,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_4peaks_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        print("4peaks_MI:", i)
        problem_list = generate_prob_4peaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                               pop_size=100,
                                                               keep_pct=0.5,
                                                               max_attempts=10,
                                                               max_iters=10,
                                                               curve=False,
                                                               random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_Cpeaks_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_Cpeaks(random_seed=0, length = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_Cpeaks_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_Cpeaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=50,
                                                                  max_iters=50,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_Cpeaks_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        problem_list = generate_prob_Cpeaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=50,
                                                          mutation_prob=0.1,
                                                          max_attempts=10,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_Cpeaks_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 100, 10):
        print("cpeask_mi:", i)
        problem_list = generate_prob_Cpeaks(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                               pop_size=100,
                                                               keep_pct=0.5,
                                                               max_attempts=10,
                                                               max_iters=10,
                                                               curve=False,
                                                               random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_TSP_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_TSP(random_seed=0, citys = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=100,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_TSP_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_TSP(random_seed=0, citys=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=1,
                                                                  max_iters=800,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_TSP_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_TSP(random_seed=0, citys=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=100,
                                                          mutation_prob=0.1,
                                                          max_attempts=10,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_TSP_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        print("TSP_MI- city size:", i)
        problem_list = generate_prob_TSP(random_seed=0, citys=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            try:
                start_time = time.time()
                best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                                   pop_size=100,
                                                                   keep_pct=0.1,
                                                                   max_attempts=10,
                                                                   max_iters=10,
                                                                   curve=False,
                                                                   random_state=1)

                time_diff = time.time() - start_time
                print("time_diff:", time_diff)
                print("best_fitness:", best_fitness)
            except:
                print("error on MI")
                best_state = np.nan
                best_fitness = np.nan
                time_diff = np.nan
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.nanmean(time_list)
        best_fitness_mean = np.nanmean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_MKC_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_MKC(random_seed=0, nodes = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_MKC_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_MKC(random_seed=0, nodes=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=50,
                                                                  max_iters=400,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_MKC_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        problem_list = generate_prob_MKC(random_seed=0, nodes=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=100,
                                                          mutation_prob=0.1,
                                                          max_attempts=10,
                                                          max_iters=10,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_MKC_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(2, 100, 5):
        print("MKC_MI- city size:", i)
        problem_list = generate_prob_MKC(random_seed=0, nodes=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            try:
                start_time = time.time()
                best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                                   pop_size=100,
                                                                   keep_pct=0.1,
                                                                   max_attempts=10,
                                                                   max_iters=10,
                                                                   curve=False,
                                                                   random_state=1)

                time_diff = time.time() - start_time
                print("time_diff:", time_diff)
                print("best_fitness:", best_fitness)
            except:
                print("error on MI")
                best_state = np.nan
                best_fitness = np.nan
                time_diff = np.nan
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.nanmean(time_list)
        best_fitness_mean = np.nanmean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_FlipFlop_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_FlipFlop(random_seed=0, length = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_FlipFlop_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_FlipFlop(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=10,
                                                                  max_iters=20,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_FlipFlop_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_FlipFlop(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=100,
                                                          mutation_prob=0.5,
                                                          max_attempts=50,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_FlipFlop_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        print("flipflop_mi:", i)
        problem_list = generate_prob_FlipFlop(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                               pop_size=50,
                                                               keep_pct=0.1,
                                                               max_attempts=10,
                                                               max_iters=10,
                                                               curve=False,
                                                               random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])

def size_test_KnapSack_RHC():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_KnapSack(random_seed=0, length = i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                                max_attempts=50,
                                                                max_iters=50,
                                                                restarts=0,
                                                                init_state=None,
                                                                curve=False,
                                                                random_state=1)
            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_KnapSack_SA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_KnapSack(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=mlrose.ExpDecay(),
                                                                  max_attempts=10,
                                                                  max_iters=20,
                                                                  random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_KnapSack_GA():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        problem_list = generate_prob_KnapSack(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          pop_size=100,
                                                          mutation_prob=0.5,
                                                          max_attempts=50,
                                                          max_iters=50,
                                                          curve=False,
                                                          random_state=0)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])
def size_test_KnapSack_MI():
    time_mean_list = []
    best_fitness_mean_list = []
    alter_list = []
    for i in range(1, 300, 30):
        print("knapsack_mi:", i)
        problem_list = generate_prob_KnapSack(random_seed=0, length=i, sample_size=5)
        time_list = []
        best_fitness_list = []
        # print(problem_list)
        for problem in problem_list:
            start_time = time.time()
            best_state, best_fitness = mlrose.algorithms.mimic(problem,
                                                               pop_size=50,
                                                               keep_pct=0.1,
                                                               max_attempts=10,
                                                               max_iters=10,
                                                               curve=False,
                                                               random_state=1)

            time_diff = time.time() - start_time
            print("time_diff:", time_diff)
            print("best_fitness:", best_fitness)
            time_list.append(time_diff)
            best_fitness_list.append(best_fitness)

        time_mean = np.mean(time_list)
        best_fitness_mean = np.mean(best_fitness_list)
        time_mean_list.append(time_mean)
        best_fitness_mean_list.append(best_fitness_mean)
        alter_list.append(i)

    return np.array([[alter_list, time_mean_list, "sample size", "clock time"],
                     [alter_list, best_fitness_mean_list, "sample_size", "best_fitness"]])


if __name__=="__main__":

    # plotting(prob_name="OneMax", algo_name="RHC", valuess = size_test_OneMax_RHC(), size_plot=True)
    # plotting(prob_name="OneMax", algo_name="SA", valuess=size_test_OneMax_SA(), size_plot=True)
    # plotting(prob_name="OneMax", algo_name="GA", valuess=size_test_OneMax_GA(), size_plot=True)
    # plotting(prob_name="OneMax", algo_name="MI", valuess=size_test_OneMax_MI(), size_plot=True)  # neet to rerun this

    plotting(prob_name="Cpeaks", algo_name="RHC", valuess=size_test_Cpeaks_RHC(), size_plot=True)
    plotting(prob_name="Cpeaks", algo_name="SA", valuess=size_test_Cpeaks_SA(), size_plot=True)
    plotting(prob_name="Cpeaks", algo_name="GA", valuess=size_test_Cpeaks_GA(), size_plot=True)
    plotting(prob_name="Cpeaks", algo_name="MI", valuess=size_test_Cpeaks_MI(), size_plot=True)
    #
    plotting(prob_name="4peaks", algo_name="RHC", valuess=size_test_4peaks_RHC(), size_plot=True)
    plotting(prob_name="4peaks", algo_name="SA", valuess=size_test_4peaks_SA(), size_plot=True)
    plotting(prob_name="4peaks", algo_name="GA", valuess=size_test_4peaks_GA(), size_plot=True)
    plotting(prob_name="4peaks", algo_name="MI", valuess=size_test_4peaks_MI(), size_plot=True)  # neet to rerun this

    # plotting(prob_name="TSP", algo_name="RHC", valuess=size_test_TSP_RHC(), size_plot=True)
    # plotting(prob_name="TSP", algo_name="SA", valuess=size_test_TSP_SA(), size_plot=True)
    # plotting(prob_name="TSP", algo_name="GA", valuess=size_test_TSP_GA(), size_plot=True)
    # plotting(prob_name="TSP", algo_name="MI", valuess=size_test_TSP_MI(), size_plot=True)
    #
    # plotting(prob_name="MKC", algo_name="RHC", valuess=size_test_MKC_RHC(), size_plot=True)
    # plotting(prob_name="MKC", algo_name="SA", valuess=size_test_MKC_SA(), size_plot=True)
    # plotting(prob_name="MKC", algo_name="GA", valuess=size_test_MKC_GA(), size_plot=True)
    # plotting(prob_name="MKC", algo_name="MI", valuess=size_test_MKC_MI(), size_plot=True)
    #
    plotting(prob_name="FlipFlop", algo_name="RHC", valuess=size_test_FlipFlop_RHC(), size_plot=True)
    plotting(prob_name="FlipFlop", algo_name="SA", valuess=size_test_FlipFlop_SA(), size_plot=True)
    plotting(prob_name="FlipFlop", algo_name="GA", valuess=size_test_FlipFlop_GA(), size_plot=True)
    plotting(prob_name="FlipFlop", algo_name="MI", valuess=size_test_FlipFlop_MI(), size_plot=True)

    plotting(prob_name="KnapSack", algo_name="RHC", valuess=size_test_KnapSack_RHC(), size_plot=True)
    plotting(prob_name="KnapSack", algo_name="SA", valuess=size_test_KnapSack_SA(), size_plot=True)
    plotting(prob_name="KnapSack", algo_name="GA", valuess=size_test_KnapSack_GA(), size_plot=True)
    plotting(prob_name="KnapSack", algo_name="MI", valuess=size_test_KnapSack_MI(), size_plot=True)

    # '''Problem 1: n-city TSP: over a map of given size,
    # generate N cities for a salesman to travel through each city and find the shortest route'''
    # problem_list_TSP = generate_prob_TSP(random_seed=0, citys=6, sample_size=5)
    # prob_to_curves("nCityTSP", problem_list_TSP)
    #
    # '''Problem 2: Max-k color optimization problem. Evaluates the fitness of an n-dimensional state vector
    # 𝑥 = [𝑥0, 𝑥1, . . . , 𝑥𝑛−1], where 𝑥𝑖 represents the color of node i, as the number of pairs of adjacent nodes of the
    # same color.'''
    # problem_list_MKC = generate_prob_MKC(random_seed=0, nodes=8, sample_size=5)
    # prob_to_curves("MaxKColor", problem_list_MKC)

    # '''Problem 3: 4 peaks
    # # Create list of states with random t_pct'''
    # problem_list_4peaks = generate_prob_4peaks(random_seed=0, length=10, sample_size=5)
    # prob_to_curves("4peaks", problem_list_4peaks)

    # '''Problem 4: OneMax
    # # Create an array made by 0 and 1, with a length between 10 and 50'''
    # problem_list_OneMax = generate_prob_OneMax(random_seed=0, length=10, sample_size= 5)
    # prob_to_curves("OneMax", problem_list_OneMax)

    '''plot time and fitness score over sample size on four algorithms'''
    '''Problem 4: OneMax:alteration of length'''






