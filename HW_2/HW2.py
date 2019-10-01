import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mlrose
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

A: Yeah, i would pro do that. u can start with like w/e setting of the problem, find to make these methods better, just stick w/ them. 
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





