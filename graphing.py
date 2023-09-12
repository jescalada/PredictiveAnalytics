# importing the required module
import matplotlib.pyplot as plt
import numpy as np


def line(x: int) -> int:
    return 3*x + 2


points = [(x, line(x)) for x in range(-2, 5)]

# unzip the points to plot
x, y = zip(*points)
plt.plot(x, y)

# draw x and y axis
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# naming the x axis
plt.xlabel('x')
# naming the y axis
plt.ylabel('y')

# giving a title to my graph
plt.title('Exercise 7')

# function to show the plot
plt.show()


# Create a polynomial function.
def polynomial(x: float) -> float:
    return 2*x**2 + 3*x + 2


# Create a complicated sine function
def sine(x: float) -> float:
    return 3*np.sin(x) + 2*np.cos(x)


# Create a function that plots a line of length 1 from the origin with a certain angle
def line(angle: float):
    # Calculate the x and y coordinates of the line by converting the angle to radians
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))

    # Plot the line
    plt.plot([0, x], [0, y])


# Create the points using both functions, then plot them in different colours
points = [(x, polynomial(x)) for x in np.arange(-2, 5, 0.1)]
x, y = zip(*points)
plt.plot(x, y, 'r')

points = [(x, sine(x)) for x in np.arange(-2, 5, 0.1)]
x, y = zip(*points)
plt.plot(x, y, 'b')

plt.show()


# Create a function that represents a circle
def circle(x: float) -> float:
    return np.sqrt(1 - x**2)

# Create the points using the circle function, then plot them
points = [(x, circle(x)) for x in np.arange(-1, 1, 0.01)]

# Add the negative half of the circle
points += [(-x, -y) for x, y in points]

# Create points for a sine function and plot them in green
points2 = [(x, np.sin(x)) for x in np.arange(-np.pi, np.pi, 0.01)]
x, y = zip(*points2)
plt.plot(x, y, 'g')

line(30)
line(60)

# Unzip the points and plot them
x, y = zip(*points)
plt.plot(x, y, 'r')

# Add the axes
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# Add the title and labels
plt.title('Exercise 9')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()