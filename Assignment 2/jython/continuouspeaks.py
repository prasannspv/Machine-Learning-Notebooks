import sys
sys.path.append("ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction

from array import array
import time
import csv

N = 50
T = N/10
ranges = array('i', [2]*N)

num_iterations = [1000*i for i in range(100)]
data = []
# No piecewise evaluation function. Run the algorithms for a range of iterations.
for iterations in num_iterations:
    print("Iterations - {}".format(iterations))
    record = [iterations]
    cp = ContinuousPeaksEvaluationFunction(T)
    unif = DiscreteUniformDistribution(ranges)
    neighbor = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(cp, unif, neighbor)
    rhc = RandomizedHillClimbing(hcp)
    model = FixedIterationTrainer(rhc, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(cp.value(rhc.getOptimal()))
    record.append(toc - tick)

    sa = SimulatedAnnealing(1e10, 0.95, hcp)
    model = FixedIterationTrainer(sa, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(cp.value(sa.getOptimal()))
    record.append(toc - tick)

    mutation = DiscreteChangeOneMutation(ranges)
    unix = SingleCrossOver()
    ggap = GenericGeneticAlgorithmProblem(cp, unif, mutation, unix)
    ga = StandardGeneticAlgorithm(200, 150, 25, ggap)
    model = FixedIterationTrainer(ga, iterations)
    model.train()
    toc = time.time()
    record.append(cp.value(ga.getOptimal()))
    record.append(toc - tick)

    tree = DiscreteDependencyTree(0.1, ranges)
    gpop = GenericProbabilisticOptimizationProblem(cp, unif, tree)
    mimic = MIMIC(200, 100, gpop)
    model = FixedIterationTrainer(mimic, iterations)
    model.train()
    toc = time.time()
    record.append(cp.value(mimic.getOptimal()))
    record.append(toc - tick)

    data.append(record)

with open("cp.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(data)
