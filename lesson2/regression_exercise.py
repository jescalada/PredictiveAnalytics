from lesson2.least_squares_regression import best_fit_line

x = [0.19, 0.28, 0.35, 0.37, 0.4, 0.18]
y = [0.13, 0.12, 0.35, 0.3, 0.37, 0.1]

# Calculate slope and y-intercept of linear regression line
m, b = best_fit_line(x, y)

# Draw line of best fit and points
import matplotlib.pyplot as plt

plt.plot(x, [m * x + b for x in x], color='red')
plt.scatter(x, y, color='green', label='Bacteria A')
plt.xlabel("Time (Days)")
plt.ylabel("Growth")
plt.legend()
plt.title("Bacteria Growth Over Time")
plt.show()
