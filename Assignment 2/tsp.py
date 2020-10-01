import sys
sys.path.append("ABAGAIL.jar")
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
from array import array
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import time
import csv

random = Random()
N = 20

points = []
for i in range(N):
    points.append([random.nextDouble(), random.nextDouble()])

num_iterations = [1000*i for i in range(100)]
data = []
# No piecewise evaluation function. Run the algorithms for a range of iterations.
for iterations in num_iterations:
    print("Iterations - {}".format(iterations))
    record = [iterations]
    tsp = TravelingSalesmanRouteEvaluationFunction(points)

    perm = DiscretePermutationDistribution(N)
    neighbor = SwapNeighbor()
    hcp = GenericHillClimbingProblem(tsp, perm, neighbor)
    rhc = RandomizedHillClimbing(hcp)
    model = FixedIterationTrainer(rhc, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(tsp.value(rhc.getOptimal()))
    record.append(toc - tick)

    sa = SimulatedAnnealing(1e10, 0.999, hcp)
    model = FixedIterationTrainer(sa, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(tsp.value(sa.getOptimal()))
    record.append(toc - tick)

    mutation = SwapMutation()
    tspx = TravelingSalesmanCrossOver(tsp)
    ggap = GenericGeneticAlgorithmProblem(tsp, perm, mutation, tspx)
    ga = StandardGeneticAlgorithm(200, 150, 25, ggap)
    model = FixedIterationTrainer(ga, iterations)
    model.train()
    toc = time.time()
    record.append(tsp.value(sa.getOptimal()))
    record.append(toc - tick)

    ranges = array('i', [N]*N)
    tree = DiscreteDependencyTree(0.1, ranges)
    gpop = GenericProbabilisticOptimizationProblem(tsp, perm, tree)
    mimic = MIMIC(200, 100, gpop)
    model = FixedIterationTrainer(mimic, iterations)
    model.train()
    toc = time.time()
    record.append(tsp.value(sa.getOptimal()))
    record.append(toc - tick)

    data.append(record)

with open("tsp.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(data)
