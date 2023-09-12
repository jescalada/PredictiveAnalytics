import matplotlib.pyplot as plt
import numpy as np

x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]

plt.bar(x, energy, color='green')
plt.xlabel("Energy Source")
plt.ylabel("Energy Output (GJ)")
plt.title("Energy output from various fuel sources")

plt.xticks(x, x)
plt.show()


NUM_MEANS     = 4
NUM_GROUPS    = 3
bc_means      = [20, 35, 30, 35, 27]
alberta_means = [25, 32, 34, 20, 25]
saskatchewan_means = [18, 28, 32, 24, 31]

ind = np.arange(NUM_MEANS)
print(ind)
width = 0.2
plt.bar(ind - width, bc_means[:-1], width, label='BC')
plt.bar(ind, alberta_means[:-1], width, label='AB')
plt.bar(ind + width, saskatchewan_means[:-1], width, label='SK')

plt.ylabel('Revenue')
plt.title('Quarterly Revenue by Province')

plt.xticks(ind + width / NUM_GROUPS, ('Q1', 'Q2', 'Q3', 'Q4'))
plt.legend(loc='best')
plt.show()
