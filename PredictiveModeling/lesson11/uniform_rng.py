import numpy as np
import matplotlib.pyplot as plt
LOW             = 0
HIGH            = 24
SIZE            = 100
NUM_SIMULATIONS = 5

bankruptcies = []

plt.subplots(nrows=1, ncols=5,  figsize=(14,7))
for i in range(1, NUM_SIMULATIONS + 1):
    # Randomize data
    x = np.random.uniform(LOW, HIGH, SIZE)
    plt.subplot(1, 5, i)
    plt.hist(x, 6, density=True)

    # Calculate percentage that score either 0 or 1.
    bankrupt = 0
    for j in range(0, SIZE):
        if x[j] < 1:
            bankrupt += 1
    bankruptcies.append(bankrupt)
plt.show()

print("Bankruptcies: " + str(bankruptcies))
print("Average bankruptcies: " + str(np.mean(bankruptcies)))
print("Standard deviation of bankruptcies: " + str(np.std(bankruptcies)))