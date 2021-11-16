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
nos = 20
knapsack_vol = max_wv * nos * duplicates * 0.4
for i in range(nos):
    weights.append(random.nextDouble() * max_wv)
    volumes.append(random.nextDouble() * max_wv)
weights = array('d', weights)
volumes = array('d', volumes)

copies = array('i', [duplicates] * nos)
ranges = array('i', [duplicates + 1] * nos)

num_iterations = [200*i for i in range(100)]


def perform_analysis():
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

        sa = SimulatedAnnealing(1e10, 0.99, hcp)
        model = FixedIterationTrainer(sa, iterations)
        tick = time.time()
        model.train()
        toc = time.time()
        record.append(knapsack.value(sa.getOptimal()))
        record.append(toc - tick)

        mutation = DiscreteChangeOneMutation(ranges)
        unix = UniformCrossOver()
        ggap = GenericGeneticAlgorithmProblem(knapsack, unif, mutation, unix)
        ga = StandardGeneticAlgorithm(200, 50, 25, ggap)
        model = FixedIterationTrainer(ga, iterations)
        tick = time.time()
        model.train()
        toc = time.time()
        record.append(knapsack.value(ga.getOptimal()))
        record.append(toc - tick)

        tree = DiscreteDependencyTree(0.1, ranges)
        gpop = GenericProbabilisticOptimizationProblem(knapsack, unif, tree)
        mimic = MIMIC(200, 100, gpop)
        model = FixedIterationTrainer(mimic, iterations)
        tick = time.time()
        model.train()
        toc = time.time()
        record.append(knapsack.value(mimic.getOptimal()))
        record.append(toc - tick)

        data.append(record)
    with open("data/knapsack.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(data)



def mimic_analysis():
    print("MIMIC")
    params = [25, 50, 100]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        knapsack = KnapsackEvaluationFunction(weights, volumes, knapsack_vol, copies)
        for toKeep in params:
            tree = DiscreteDependencyTree(0.1, ranges)
            unif = DiscreteUniformDistribution(ranges)
            gpop = GenericProbabilisticOptimizationProblem(knapsack, unif, tree)
            mimic = MIMIC(200, toKeep, gpop)
            model = FixedIterationTrainer(mimic, iterations)
            model.train()
            record.append(knapsack.value(mimic.getOptimal()))
        records.append(record)
    with open("data/knapsack_MIMIC.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


def sa_analysis():
    print("SA")
    params = [0.99, 0.95, 0.75, 0.5]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        knapsack = KnapsackEvaluationFunction(weights, volumes, knapsack_vol, copies)
        for decay in params:
            neighbor = DiscreteChangeOneNeighbor(ranges)
            unif = DiscreteUniformDistribution(ranges)
            hcp = GenericHillClimbingProblem(knapsack, unif, neighbor)
            sa = SimulatedAnnealing(1e10, decay, hcp)
            model = FixedIterationTrainer(sa, iterations)
            model.train()
            record.append(knapsack.value(sa.getOptimal()))
        records.append(record)
    with open("data/knapsack_SA.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


def ga_analysis():
    print("GA")
    params = [(100, 100), (100, 50), (100, 25), (50, 100), (50, 50), (50, 25)]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        knapsack = KnapsackEvaluationFunction(weights, volumes, knapsack_vol, copies)
        unif = DiscreteUniformDistribution(ranges)
        mutation = DiscreteChangeOneMutation(ranges)
        unix = UniformCrossOver()
        ggap = GenericGeneticAlgorithmProblem(knapsack, unif, mutation, unix)
        for to_mate, to_mutate in params:
            ga = StandardGeneticAlgorithm(200, to_mate, to_mutate, ggap)
            model = FixedIterationTrainer(ga, iterations)
            model.train()
            record.append(knapsack.value(ga.getOptimal()))
        records.append(record)
    with open("data/knapsack_GA.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


# ga_analysis()
# sa_analysis()
# mimic_analysis()
perform_analysis()
