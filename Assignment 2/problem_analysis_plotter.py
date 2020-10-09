import matplotlib.pyplot as plt
import csv


def plotter(problem):
    x = []
    ga = [[] for i in range(6)]
    with open(f"jython/data/{problem}_GA.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[0]))
            for i in range(1, 7):
                ga[i-1].append(float(row[i]))

    label = [f"to_mate={to_mate},to_mutate={to_mutate}"
             for to_mate, to_mutate in [(100, 100), (100, 50), (100, 25), (50, 100), (50, 50), (50, 25)]]
    for i in range(6):
        plt.plot(x, ga[i], label=label[i])
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.savefig(f"img/{problem}_GA.png")
    plt.clf()

    sa = [[] for i in range(4)]
    with open(f"jython/data/{problem}_SA.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, 5):
                sa[i-1].append(float(row[i]))

    label = [f"decay={decay}"
             for decay in [0.99, 0.95, 0.75, 0.5]]
    for i in range(4):
        plt.plot(x, sa[i], label=label[i])
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc = "best")
    plt.savefig(f"img/{problem}_SA.png")
    plt.clf()

    mimic = [[] for i in range(3)]
    with open(f"jython/data/{problem}_MIMIC.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, 4):
                mimic[i-1].append(float(row[i]))

    label = [f"tokeep={tokeep}"
             for tokeep in [25, 50, 100]]
    for i in range(3):
        plt.plot(x, mimic[i], label=label[i])
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc = "best")
    plt.savefig(f"img/{problem}_MIMIC.png")


# plotter("cp")
# plotter("tsp")
plotter("knapsack")