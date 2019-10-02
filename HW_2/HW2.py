import numpy as np
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


def nCityTSP(city_num=10, random_seed = 0):
    '''Problem 1: n-city TSP: over a map of given size,
    generate N cities for a salesman to travel through each city and find the shortest route'''
    # Create list of city coordinates
    np.random.seed(random_seed)
    coords_list = []
    for i in range(city_num):
        coords_list.append((np.random.random_sample(), np.random.random_sample()))

    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords=coords_list)

    # Define optimization problem object
    problem = mlrose.TSPOpt(length=city_num, fitness_fn=fitness_coords, maximize=False)

    "========random hill climbing========"
    best_fitness_RHC_list = []
    best_fitness_RHC_att_list = []
    alter_list_RHC = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("RHC:", i)
        best_state_RHC, best_fitness_RHC = mlrose.random_hill_climb(problem,
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    restarts=0,
                                                                    init_state=None,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_RHC_att, best_fitness_RHC_att = mlrose.random_hill_climb(problem,
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            restarts=0,
                                                                            init_state=None,
                                                                            curve=False,
                                                                            random_state=1)
        best_fitness_RHC_list.append(best_fitness_RHC)
        best_fitness_RHC_att_list.append(best_fitness_RHC_att)
        alter_list_RHC.append(i)

    # plotting

    plt.plot(alter_list_RHC, best_fitness_RHC_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('RHC_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_RHC, best_fitness_RHC_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('RHC_max_attempt.png')
    plt.gcf().clear()

    "========simulated annealing========"
    best_fitness_SA_list = []
    best_fitness_SA_att_list = []
    alter_list_SA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("SA:", i)
        best_state_SA, best_fitness_SA = mlrose.simulated_annealing(problem,
                                                                    schedule=mlrose.ExpDecay(),
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    random_state=1)
        best_state_SA_att, best_fitness_SA_att = mlrose.simulated_annealing(problem,
                                                                            schedule=mlrose.ExpDecay(),
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            random_state=1)
        best_fitness_SA_list.append(best_fitness_SA)
        best_fitness_SA_att_list.append(best_fitness_SA_att)
        alter_list_SA.append(i)
    # return alter_list_SA, best_fitness_SA_list, best_fitness_SA_att_list
    # plotting
    plt.grid()

    plt.plot(alter_list_SA, best_fitness_SA_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('SA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_SA, best_fitness_SA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('SA_max_attempt.png')
    plt.gcf().clear()

    "========GA========"
    best_fitness_GA_list = []
    best_fitness_GA_att_list = []
    best_fitness_GA_pop_list = []
    best_fitness_GA_mutpb_list = []
    alter_list_GA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("GA:", i)
        start_time = time.time()
        best_state_GA, best_fitness_GA = mlrose.genetic_alg(problem,
                                                            pop_size=200,
                                                            mutation_prob=0.1,
                                                            max_attempts=1000,
                                                            max_iters=i,
                                                            curve=False,
                                                            random_state=0)
        best_state_GA_att, best_fitness_GA_att = mlrose.genetic_alg(problem,
                                                                    pop_size=200,
                                                                    mutation_prob=0.1,
                                                                    max_attempts=i,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=0)
        best_state_GA_pop, best_fitness_GA_pop = mlrose.genetic_alg(problem,
                                                                    pop_size=max(1, int(i / 5)),
                                                                    mutation_prob=0.1,
                                                                    max_attempts=1000,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_GA_mutpb, best_fitness_GA_mutpb = mlrose.genetic_alg(problem,
                                                                        pop_size=200,
                                                                        mutation_prob=i/1000,
                                                                        max_attempts=1000,
                                                                        max_iters=1000,
                                                                        curve=False,
                                                                        random_state=1)
        best_fitness_GA_list.append(best_fitness_GA)
        best_fitness_GA_att_list.append(best_fitness_GA_att)
        best_fitness_GA_pop_list.append(best_fitness_GA_pop)
        best_fitness_GA_mutpb_list.append(best_fitness_GA_mutpb)
        alter_list_GA.append(i)
        end_time = time.time()
        print(end_time - start_time)
    # plotting
    plt.grid()

    plt.plot(alter_list_GA, best_fitness_MI_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('GA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_GA, best_fitness_GA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('GA_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_GA), best_fitness_GA_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('GA_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_GA), best_fitness_GA_mutpb_list, color="r",
             # label="max_iters"
             )
    x_title = "mutation_prob"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('GA_mutation_prob.png')
    plt.gcf().clear()

    "========MIMIC========"
    best_fitness_MI_list = []
    best_fitness_MI_att_list = []
    best_fitness_MI_pop_list = []
    best_fitness_MI_pct_list = []
    alter_list_MI = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("MI:", i)
        start_time = time.time()
        best_state_MI, best_fitness_MI = mlrose.algorithms.mimic(problem,
                                                                 pop_size=200,
                                                                 keep_pct=0.2,
                                                                 max_attempts=1000,
                                                                 max_iters=i,
                                                                 curve=False,
                                                                 random_state = 1)
        best_state_MI_att, best_fitness_MI_att = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=0.2,
                                                                         max_attempts=i,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state = 1)
        best_state_MI_pop, best_fitness_MI_pop = mlrose.algorithms.mimic(problem,
                                                                         pop_size=int(i/5),
                                                                         keep_pct=0.2,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_state_MI_pct, best_fitness_MI_pct = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=float(i)/1000,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_fitness_MI_list.append(best_fitness_MI)
        best_fitness_MI_att_list.append(best_fitness_MI_att)
        best_fitness_MI_pop_list.append(best_fitness_MI_pop)
        best_fitness_MI_pct_list.append(best_fitness_MI_pct)
        alter_list_MI.append(i)
        end_time = time.time()
        print(end_time-start_time)
    # plotting

    plt.grid()

    plt.plot(alter_list, best_fitness_MI_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MI_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list, best_fitness_MI_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MI_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_MI), best_fitness_MI_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MI_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_MI), best_fitness_MI_pct_list, color="r",
             # label="max_iters"
             )
    x_title = "keep_pct"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MI_pct_size.png')
    plt.gcf().clear()


def MaxKColor(nodes=8, random_seed = 0):
    '''Problem 2: Max-k color optimization problem. Evaluates the fitness of an n-dimensional state vector
𝑥 = [𝑥0, 𝑥1, . . . , 𝑥𝑛−1], where 𝑥𝑖 represents the color of node i, as the number of pairs of adjacent nodes of the
same color.'''
    # Create list of city coordinates
    np.random.seed(random_seed)
    edges_list = []
    for i in range(nodes):
        for j in range(i+1, nodes):
            edges_list.append((i, j))
    # print(edges_list)

    # Initialize fitness function object using edges_list
    fitness = mlrose.MaxKColor(edges_list)

    # Define optimization problem object
    # problem = mlrose.TSPOpt(length=city_num, fitness_fn=fitness_coords, maximize=False)
    problem = mlrose.DiscreteOpt(length = nodes, fitness_fn = fitness, maximize=False)

    "========random hill climbing========"
    best_fitness_RHC_list = []
    best_fitness_RHC_att_list = []
    alter_list_RHC = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("RHC:", i)
        best_state_RHC, best_fitness_RHC = mlrose.random_hill_climb(problem,
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    restarts=0,
                                                                    init_state=None,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_RHC_att, best_fitness_RHC_att = mlrose.random_hill_climb(problem,
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            restarts=0,
                                                                            init_state=None,
                                                                            curve=False,
                                                                            random_state=1)
        best_fitness_RHC_list.append(best_fitness_RHC)
        best_fitness_RHC_att_list.append(best_fitness_RHC_att)
        alter_list_RHC.append(i)

    # plotting

    plt.plot(alter_list_RHC, best_fitness_RHC_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('MKC_RHC_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_RHC, best_fitness_RHC_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('MKC_RHC_max_attempt.png')
    plt.gcf().clear()

    "========simulated annealing========"
    best_fitness_SA_list = []
    best_fitness_SA_att_list = []
    alter_list_SA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("SA:", i)
        best_state_SA, best_fitness_SA = mlrose.simulated_annealing(problem,
                                                                    schedule=mlrose.ExpDecay(),
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    random_state=1)
        best_state_SA_att, best_fitness_SA_att = mlrose.simulated_annealing(problem,
                                                                            schedule=mlrose.ExpDecay(),
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            random_state=1)
        best_fitness_SA_list.append(best_fitness_SA)
        best_fitness_SA_att_list.append(best_fitness_SA_att)
        alter_list_SA.append(i)
    # return alter_list_SA, best_fitness_SA_list, best_fitness_SA_att_list
    # plotting
    plt.grid()

    plt.plot(alter_list_SA, best_fitness_SA_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('MKC_SA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_SA, best_fitness_SA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('MKC_SA_max_attempt.png')
    plt.gcf().clear()

    "========GA========"
    best_fitness_GA_list = []
    best_fitness_GA_att_list = []
    best_fitness_GA_pop_list = []
    best_fitness_GA_mutpb_list = []
    alter_list_GA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("GA:", i)
        start_time = time.time()
        best_state_GA, best_fitness_GA = mlrose.genetic_alg(problem,
                                                            pop_size=200,
                                                            mutation_prob=0.1,
                                                            max_attempts=1000,
                                                            max_iters=i,
                                                            curve=False,
                                                            random_state=0)
        best_state_GA_att, best_fitness_GA_att = mlrose.genetic_alg(problem,
                                                                    pop_size=200,
                                                                    mutation_prob=0.1,
                                                                    max_attempts=i,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=0)
        best_state_GA_pop, best_fitness_GA_pop = mlrose.genetic_alg(problem,
                                                                    pop_size=max(1, int(i / 5)),
                                                                    mutation_prob=0.1,
                                                                    max_attempts=1000,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_GA_mutpb, best_fitness_GA_mutpb = mlrose.genetic_alg(problem,
                                                                        pop_size=200,
                                                                        mutation_prob=i/1000,
                                                                        max_attempts=1000,
                                                                        max_iters=1000,
                                                                        curve=False,
                                                                        random_state=1)
        best_fitness_GA_list.append(best_fitness_GA)
        best_fitness_GA_att_list.append(best_fitness_GA_att)
        best_fitness_GA_pop_list.append(best_fitness_GA_pop)
        best_fitness_GA_mutpb_list.append(best_fitness_GA_mutpb)
        alter_list_GA.append(i)
        end_time = time.time()
        print(end_time - start_time)
    # plotting
    plt.grid()

    plt.plot(alter_list_GA, best_fitness_GA_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_GA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_GA, best_fitness_GA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_GA_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_GA), best_fitness_GA_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_GA_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_GA), best_fitness_GA_mutpb_list, color="r",
             # label="max_iters"
             )
    x_title = "mutation_prob"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_GA_mutation_prob.png')
    plt.gcf().clear()

    "========MIMIC========"
    best_fitness_MI_list = []
    best_fitness_MI_att_list = []
    best_fitness_MI_pop_list = []
    best_fitness_MI_pct_list = []
    alter_list_MI = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("MI:", i)
        start_time = time.time()
        best_state_MI, best_fitness_MI = mlrose.algorithms.mimic(problem,
                                                                 pop_size=200,
                                                                 keep_pct=0.2,
                                                                 max_attempts=1000,
                                                                 max_iters=i,
                                                                 curve=False,
                                                                 random_state = 1)
        best_state_MI_att, best_fitness_MI_att = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=0.2,
                                                                         max_attempts=i,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state = 1)
        best_state_MI_pop, best_fitness_MI_pop = mlrose.algorithms.mimic(problem,
                                                                         pop_size=int(i/5),
                                                                         keep_pct=0.2,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_state_MI_pct, best_fitness_MI_pct = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=float(i)/1000,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_fitness_MI_list.append(best_fitness_MI)
        best_fitness_MI_att_list.append(best_fitness_MI_att)
        best_fitness_MI_pop_list.append(best_fitness_MI_pop)
        best_fitness_MI_pct_list.append(best_fitness_MI_pct)
        alter_list_MI.append(i)
        end_time = time.time()
        print(end_time-start_time)
    # plotting

    plt.grid()

    plt.plot(alter_list, best_fitness_MI_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_MI_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list, best_fitness_MI_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_MI_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_MI), best_fitness_MI_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_MI_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_MI), best_fitness_MI_pct_list, color="r",
             # label="max_iters"
             )
    x_title = "keep_pct"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('MKC_MI_pct_size.png')
    plt.gcf().clear()

def four_peaks(random_seed = 0):
    '''Problem 3: Max-k color optimization problem. Evaluates the fitness of an n-dimensional state vector
𝑥 = [𝑥0, 𝑥1, . . . , 𝑥𝑛−1], where 𝑥𝑖 represents the color of node i, as the number of pairs of adjacent nodes of the
same color.'''
    # Create list of city coordinates
    np.random.seed(random_seed)
    state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    # print(edges_list)

    # Initialize fitness function object using edges_list
    fitness = mlrose.FourPeaks(t_pct=0.15)

    # Define optimization problem object
    # problem = mlrose.TSPOpt(length=city_num, fitness_fn=fitness_coords, maximize=False)
    problem = mlrose.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize=True, max_val = 2)

    "========random hill climbing========"
    best_fitness_RHC_list = []
    best_fitness_RHC_att_list = []
    alter_list_RHC = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("RHC:", i)
        best_state_RHC, best_fitness_RHC = mlrose.random_hill_climb(problem,
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    restarts=0,
                                                                    init_state=None,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_RHC_att, best_fitness_RHC_att = mlrose.random_hill_climb(problem,
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            restarts=0,
                                                                            init_state=None,
                                                                            curve=False,
                                                                            random_state=1)
        best_fitness_RHC_list.append(best_fitness_RHC)
        best_fitness_RHC_att_list.append(best_fitness_RHC_att)
        alter_list_RHC.append(i)

    # plotting

    plt.plot(alter_list_RHC, best_fitness_RHC_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('4p_RHC_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_RHC, best_fitness_RHC_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('4p_RHC_max_attempt.png')
    plt.gcf().clear()

    "========simulated annealing========"
    best_fitness_SA_list = []
    best_fitness_SA_att_list = []
    alter_list_SA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("SA:", i)
        best_state_SA, best_fitness_SA = mlrose.simulated_annealing(problem,
                                                                    schedule=mlrose.ExpDecay(),
                                                                    max_attempts=1000,
                                                                    max_iters=i,
                                                                    random_state=1)
        best_state_SA_att, best_fitness_SA_att = mlrose.simulated_annealing(problem,
                                                                            schedule=mlrose.ExpDecay(),
                                                                            max_attempts=i,
                                                                            max_iters=1000,
                                                                            random_state=1)
        best_fitness_SA_list.append(best_fitness_SA)
        best_fitness_SA_att_list.append(best_fitness_SA_att)
        alter_list_SA.append(i)
    # return alter_list_SA, best_fitness_SA_list, best_fitness_SA_att_list
    # plotting
    plt.grid()

    plt.plot(alter_list_SA, best_fitness_SA_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('4p_SA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_SA, best_fitness_SA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc="best")
    plt.savefig('4p_SA_max_attempt.png')
    plt.gcf().clear()

    "========GA========"
    best_fitness_GA_list = []
    best_fitness_GA_att_list = []
    best_fitness_GA_pop_list = []
    best_fitness_GA_mutpb_list = []
    alter_list_GA = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("GA:", i)
        start_time = time.time()
        best_state_GA, best_fitness_GA = mlrose.genetic_alg(problem,
                                                            pop_size=200,
                                                            mutation_prob=0.1,
                                                            max_attempts=1000,
                                                            max_iters=i,
                                                            curve=False,
                                                            random_state=0)
        best_state_GA_att, best_fitness_GA_att = mlrose.genetic_alg(problem,
                                                                    pop_size=200,
                                                                    mutation_prob=0.1,
                                                                    max_attempts=i,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=0)
        best_state_GA_pop, best_fitness_GA_pop = mlrose.genetic_alg(problem,
                                                                    pop_size=max(1, int(i / 5)),
                                                                    mutation_prob=0.1,
                                                                    max_attempts=1000,
                                                                    max_iters=1000,
                                                                    curve=False,
                                                                    random_state=1)
        best_state_GA_mutpb, best_fitness_GA_mutpb = mlrose.genetic_alg(problem,
                                                                        pop_size=200,
                                                                        mutation_prob=i/1000,
                                                                        max_attempts=1000,
                                                                        max_iters=1000,
                                                                        curve=False,
                                                                        random_state=1)
        best_fitness_GA_list.append(best_fitness_GA)
        best_fitness_GA_att_list.append(best_fitness_GA_att)
        best_fitness_GA_pop_list.append(best_fitness_GA_pop)
        best_fitness_GA_mutpb_list.append(best_fitness_GA_mutpb)
        alter_list_GA.append(i)
        end_time = time.time()
        print(end_time - start_time)
    # plotting
    plt.grid()

    plt.plot(alter_list_GA, best_fitness_MI_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_GA_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list_GA, best_fitness_GA_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_GA_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_GA), best_fitness_GA_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_GA_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_GA), best_fitness_GA_mutpb_list, color="r",
             # label="max_iters"
             )
    x_title = "mutation_prob"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_GA_mutation_prob.png')
    plt.gcf().clear()

    "========MIMIC========"
    best_fitness_MI_list = []
    best_fitness_MI_att_list = []
    best_fitness_MI_pop_list = []
    best_fitness_MI_pct_list = []
    alter_list_MI = []

    # Solve problem using simulated annealing
    for i in range(1, 1000, 100):
        print("MI:", i)
        start_time = time.time()
        best_state_MI, best_fitness_MI = mlrose.algorithms.mimic(problem,
                                                                 pop_size=200,
                                                                 keep_pct=0.2,
                                                                 max_attempts=1000,
                                                                 max_iters=i,
                                                                 curve=False,
                                                                 random_state = 1)
        best_state_MI_att, best_fitness_MI_att = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=0.2,
                                                                         max_attempts=i,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state = 1)
        best_state_MI_pop, best_fitness_MI_pop = mlrose.algorithms.mimic(problem,
                                                                         pop_size=int(i/5),
                                                                         keep_pct=0.2,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_state_MI_pct, best_fitness_MI_pct = mlrose.algorithms.mimic(problem,
                                                                         pop_size=200,
                                                                         keep_pct=float(i)/1000,
                                                                         max_attempts=1000,
                                                                         max_iters=1000,
                                                                         curve=False,
                                                                         random_state=1)
        best_fitness_MI_list.append(best_fitness_MI)
        best_fitness_MI_att_list.append(best_fitness_MI_att)
        best_fitness_MI_pop_list.append(best_fitness_MI_pop)
        best_fitness_MI_pct_list.append(best_fitness_MI_pct)
        alter_list_MI.append(i)
        end_time = time.time()
        print(end_time-start_time)
    # plotting

    plt.grid()

    plt.plot(alter_list, best_fitness_MI_list, color="r",
             # label="max_iters"
             )
    x_title = "max_iters"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_MI_max_iter.png')
    plt.gcf().clear()

    plt.plot(alter_list, best_fitness_MI_att_list, color="r",
             # label="max_iters"
             )
    x_title = "max_attempts"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_MI_max_attempt.png')
    plt.gcf().clear()

    plt.plot((x/5 for x in alter_list_MI), best_fitness_MI_pop_list, color="r",
             # label="max_iters"
             )
    x_title = "pop_size"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_MI_pop_size.png')
    plt.gcf().clear()

    plt.plot((x/1000 for x in alter_list_MI), best_fitness_MI_pct_list, color="r",
             # label="max_iters"
             )
    x_title = "keep_pct"
    y_title = "best_fitness"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig('4p_MI_pct_size.png')
    plt.gcf().clear()


if __name__=="__main__":
    # nCityTSP(city_num=10)
    MaxKColor(nodes=8, random_seed=0)
    four_peaks(random_seed=0)





