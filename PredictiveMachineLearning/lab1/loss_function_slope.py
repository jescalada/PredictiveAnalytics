weights = [0.5, 2.3, 2.9]
heights = [1.4, 1.9, 3.2]


def get_slope_of_loss_function(x, y, intercept):
    total = 0
    BETA = 0.64
    for i in range(0, len(x)):
        total += -2 * (y[i] - intercept - BETA * x[i])

    print("Intercept: " + str(intercept) + " Res: " + str(round(total, 2)))


sample_intercept = 0.57
get_slope_of_loss_function(weights, heights, sample_intercept)
