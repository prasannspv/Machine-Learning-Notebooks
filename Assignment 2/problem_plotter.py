import matplotlib.pyplot as plt
import csv


def plot(problem):
    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness function")
    x = []

    ga = []
    sa = []
    rhc = []
    mimic = []

    ga_t = []
    sa_t = []
    rhc_t = []
    mimic_t = []
    with open("jython/{}.csv".format(problem)) as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[0]))
            rhc.append(float(row[1]))
            rhc_t.append(float(row[2]))
            sa.append(float(row[3]))
            sa_t.append(float(row[4]))
            ga.append(float(row[5]))
            ga_t.append(float(row[6]))
            mimic.append(float(row[7]))
            mimic_t.append(float(row[8]))

    ax.plot(x, rhc, label="RHC")
    ax.plot(x, ga, label="GA")
    ax.plot(x, mimic, label="MIMIC")
    ax.plot(x, sa, label="SA")
    ax.legend(loc="best")

    plt.savefig("img/{}.png".format(problem))

    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Time Taken")
    ax.plot(x, rhc_t, label = "RHC")
    ax.plot(x, ga_t, label = "GA")
    ax.plot(x, mimic_t, label = "MIMIC")
    ax.plot(x, sa_t, label = "SA")
    ax.legend(loc = "best")

    plt.savefig("img/{}-time.png".format(problem))



# plot("knapsack")
# plot("tsp")
plot("cp")