import matplotlib.pyplot as plt
import csv

x = []
val = []
with open("neural_network/GA_MP.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Mating Probability")
ax.set_ylabel("Accuracy")
ax.plot(x, val)
plt.savefig("img/GA_MP.png")

x = []
val = []
with open("neural_network/GA_MP_time.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Mating Probability")
ax.set_ylabel("Accuracy")
ax.plot(x, val)
plt.savefig("img/GA_MP_time.png")

x = []
val = []
with open("neural_network/GA_POP.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Initial Population")
ax.set_ylabel("Accuracy")
ax.plot(x, val)
plt.savefig("img/GA_POP.png")

x = []
val = []
with open("neural_network/GA_POP_time.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Initial Population")
ax.set_ylabel("Time")
ax.plot(x, val)
plt.savefig("img/GA_POP_time.png")


x = []
val = []
with open("neural_network/RHC_REST.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Restarts")
ax.set_ylabel("Accuracy")
ax.plot(x, val)
plt.savefig("img/RHC_REST.png")

x = []
val = []
with open("neural_network/RHC_REST_time.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Restarts")
ax.set_ylabel("Time")
ax.plot(x, val)
plt.savefig("img/RHC_REST_time.png")

x = []
val = []
with open("neural_network/SA.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Decay")
ax.set_ylabel("Accuracy")
ax.plot(x, val)
plt.savefig("img/SA.png")

x = []
val = []
with open("neural_network/SA_time.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        val.append(float(row[1]))

fig, ax = plt.subplots()
ax.set_xlabel("Decay")
ax.set_ylabel("Time")
ax.plot(x, val)
plt.savefig("img/SA_time.png")

