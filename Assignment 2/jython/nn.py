import sys
sys.path.append("ABAGAIL.jar")
import csv
import time

from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import BackPropagationNetworkFactory
from shared import Instance
from shared import DataSet
from shared import SumOfSquaresError
from opt import RandomizedHillClimbing
from opt import SimulatedAnnealing
from opt.ga import StandardGeneticAlgorithm


def train(algo_inst, nn, data):
    results = []
    for i in range(2000):
        tick = time.time()
        algo_inst.train()
        toc = time.time()
        accurate = 0
        sse = 0

        for datum in data:
            nn.setInputValues(datum.getData())
            nn.run()

            expected = datum.getLabel().getContinuous()
            actual = nn.getOutputValues().get(0)

            if abs(expected - actual) < 0.5:
                accurate += 1

            expected = datum.getLabel()
            actual = nn.getOutputValues()

            example = Instance(actual, Instance(actual.get(0)))
            sse += SumOfSquaresError().value(expected, example)

        accuracy = accurate/len(data)
        results.append([i, toc-tick, accuracy, sse])
    return results


def get_data():
    data = []
    with open("phishing.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            datum = Instance([int(val) for val in row[:-1]])
            datum.setLabel(Instance(0 if int(row[-1]) == -1 else 1))
            data.append(datum)

    return data


def genetic_algo(data, pop=200, to_mate=100, to_mutate=50):
    factory = BackPropagationNetworkFactory()
    nn = factory.createClassificationNetwork([30, 100, 2])
    problem = NeuralNetworkOptimizationProblem(DataSet(data), nn, SumOfSquaresError())
    ga = StandardGeneticAlgorithm(pop, to_mate, to_mutate, problem)
    results = train(ga, nn, data)
    with open("GA-{}-{}-{}.csv".format(pop, to_mate, to_mutate), "w+") as f:
        csv.writer(f).writerows(results)


def run_genetic_algorithm_exp():
    data = get_data()
    population = [10, 50, 100, 200, 500]
    to_mate = [50, 100, 150, 200]
    to_mutate = [25, 50, 100, 200]

    # Perform population experiments
    for pop in population:
        genetic_algo(data, pop = pop)


if __name__ == "__main__":
    run_genetic_algorithm_exp()

