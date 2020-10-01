import sys
sys.path.append("ABAGAIL.jar")
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array
import time
import csv

max_wv = 50
duplicates = 4
random = Random()
weights = []
volumes = []
nos = 40
knapsack_vol = max_wv * nos * duplicates * 0.4
for i in range(nos):
    weights.append(random.nextDouble() * max_wv)
    volumes.append(random.nextDouble() * max_wv)
weights = array('d', weights)
volumes = array('d', volumes)

copies = array('i', [duplicates] * nos)
ranges = array('i', [duplicates + 1] * nos)

num_iterations = [1000*i for i in range(100)]
data = []
# No piecewise evaluation function. Run the algorithms for a range of iterations.
for iterations in num_iterations:
    print("Iterations - {}".format(iterations))
    record = [iterations]
    knapsack = KnapsackEvaluationFunction(weights, volumes, knapsack_vol, copies)

    unif = DiscreteUniformDistribution(ranges)
    neighbor = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(knapsack, unif, neighbor)
    rhc = RandomizedHillClimbing(hcp)
    model = FixedIterationTrainer(rhc, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(knapsack.value(rhc.getOptimal()))
    record.append(toc - tick)

    sa = SimulatedAnnealing(1e10, 0.95, hcp)
    model = FixedIterationTrainer(sa, iterations)
    tick = time.time()
    model.train()
    toc = time.time()
    record.append(knapsack.value(sa.getOptimal()))
    record.append(toc - tick)

    mutation = DiscreteChangeOneMutation(ranges)
    unix = UniformCrossOver()
    ggap = GenericGeneticAlgorithmProblem(knapsack, unif, mutation, unix)
    ga = StandardGeneticAlgorithm(200, 150, 25, ggap)
    model = FixedIterationTrainer(ga, iterations)
    model.train()
    toc = time.time()
    record.append(knapsack.value(sa.getOptimal()))
    record.append(toc - tick)

    tree = DiscreteDependencyTree(0.1, ranges)
    gpop = GenericProbabilisticOptimizationProblem(knapsack, unif, tree)
    mimic = MIMIC(200, 100, gpop)
    model = FixedIterationTrainer(mimic, iterations)
    model.train()
    toc = time.time()
    record.append(knapsack.value(sa.getOptimal()))
    record.append(toc - tick)

    data.append(record)

with open("knapsack.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(data)
