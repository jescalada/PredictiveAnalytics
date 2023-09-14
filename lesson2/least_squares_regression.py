import matplotlib.pyplot as plt


def best_fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Return slope and y-intercept of linear regression line.

    :param x: list of x coordinates
    :param y: list of y coordinates
    :precondition: x and y have the same number of items.
    :postcondition: slope and y-intercept of linear regression line are returned.
    :return: slope and y-intercept of linear regression line.
    """
    n = len(x)

    # Calculate the sums.
    x_sum = sum(x)
    y_sum = sum(y)
    xx_sum = sum([xi * xi for xi in x])
    xy_sum = sum([xi * yi for xi, yi in zip(x, y)])

    # Calculate the slope and y-intercept.
    slope = (n * xy_sum - x_sum*y_sum) / (n*xx_sum - x_sum**2)
    intercept = (y_sum - slope*x_sum) / n

    return slope, intercept


days = [0.2, 0.32, 0.38, 0.41, 0.43]
bacteria = [0.1, 0.15, 0.4, 0.6, 0.44]

# Calculate slope and y-intercept of linear regression line
m, b = best_fit_line(days, bacteria)

# Draw line of best fit and points
plt.plot(days, [m * x + b for x in days], color='red')
plt.scatter(days, bacteria, color='green', label='Bacteria A')
plt.xlabel("Time (Days)")
plt.ylabel("Growth")
plt.legend()
plt.title("Bacteria Growth Over Time")
plt.show()
