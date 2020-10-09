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

num_iterations = [200*i for i in range(100)]


def performance_comparison():
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


def mimic_analysis():
    print("MIMIC")
    params = [25, 50, 100]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        cp = ContinuousPeaksEvaluationFunction(T)
        for toKeep in params:
            tree = DiscreteDependencyTree(0.1, ranges)
            unif = DiscreteUniformDistribution(ranges)
            gpop = GenericProbabilisticOptimizationProblem(cp, unif, tree)
            mimic = MIMIC(200, toKeep, gpop)
            model = FixedIterationTrainer(mimic, iterations)
            model.train()
            record.append(cp.value(mimic.getOptimal()))
        records.append(record)
    with open("data/cp_MIMIC.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


def sa_analysis():
    print("SA")
    params = [0.99, 0.95, 0.75, 0.5]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        cp = ContinuousPeaksEvaluationFunction(T)
        for decay in params:
            neighbor = DiscreteChangeOneNeighbor(ranges)
            unif = DiscreteUniformDistribution(ranges)
            hcp = GenericHillClimbingProblem(cp, unif, neighbor)
            sa = SimulatedAnnealing(1e10, decay, hcp)
            model = FixedIterationTrainer(sa, iterations)
            model.train()
            record.append(cp.value(sa.getOptimal()))
        records.append(record)
    with open("data/cp_SA.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


def ga_analysis():
    print("GA")
    params = [(100, 100), (100, 50), (100, 25), (50, 100), (50, 50), (50, 25)]
    records = []
    for iterations in num_iterations:
        print("Iteration: ", iterations)
        record = [iterations]
        cp = ContinuousPeaksEvaluationFunction(T)
        unif = DiscreteUniformDistribution(ranges)
        mutation = DiscreteChangeOneMutation(ranges)
        unix = SingleCrossOver()
        ggap = GenericGeneticAlgorithmProblem(cp, unif, mutation, unix)
        for to_mate, to_mutate in params:
            ga = StandardGeneticAlgorithm(200, to_mate, to_mutate, ggap)
            model = FixedIterationTrainer(ga, iterations)
            model.train()
            record.append(cp.value(ga.getOptimal()))
        records.append(record)
    with open("data/cp_GA.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(records)


ga_analysis()
sa_analysis()
mimic_analysis()
# performance_comparison()
