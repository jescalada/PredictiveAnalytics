import matplotlib.pyplot as plt

# Plot scatter of x and y coordinates.
time_X    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.2]
growth_Y  = [0.1, 0.15, 0.4,  0.6, 0.44, 0.17]
plt.scatter(time_X, growth_Y, color='green', label='Bacteria A')

# Add a legend, axis labels, and title.
plt.legend()
plt.xlabel("Time (Hours)")
plt.ylabel("Growth")
plt.title('Bacteria Growth Over Time')

plt.show()

pounds = [120, 110, 160]
inches = [50, 48, 68]
plt.scatter(pounds, inches, color='orange', label='Students Region A')
plt.xlabel("Weight (Pounds)")
plt.ylabel("Height (Inches)")
plt.legend()
plt.title("Height vs. Weight for Students Region A")
plt.show()

poundsA = [120, 110, 160]
inchesA = [50, 48, 68]
poundsB = [121, 108, 150, 121, 121, 146]
inchesB = [49, 45, 85, 46, 50, 85]
plt.scatter(poundsA, inchesA, color='orange', label='Students Region A')
plt.scatter(poundsB, inchesB, color='green', label='Students Region B')
plt.xlabel("Weight (Pounds)")
plt.ylabel("Height (Inches)")
plt.legend()
plt.title("Height vs. Weight for Students in Regions A and B")
plt.show()
