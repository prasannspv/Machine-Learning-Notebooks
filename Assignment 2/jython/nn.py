import sys
sys.path.append("ABAGAIL.jar")
import csv
import time

from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import BackPropagationNetworkFactory
from shared import Instance
from shared import DataSet
from shared import SumOfSquaredError
from opt import RandomizedHillClimbing
from opt import SimulatedAnnealing
from opt import StandardGeneticAlgorithm


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
            sse += SumOfSquaredError().value(expected, example)

        accuracy = accurate/len(data)
        results.append([i, toc-tick, accuracy, sse])
    return results


def get_data():
    data = []
    with open("phishing.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            datum.setLabel(Instance(0 if int(row[-1]) == -1 else 1))
            datum = Instance([int(val) for val in row[1:-1]])
            data.append(datum)

    return data

if __name__ == "__main__":
    factory = BackPropagationNetworkFactory()
    data = get_data()

