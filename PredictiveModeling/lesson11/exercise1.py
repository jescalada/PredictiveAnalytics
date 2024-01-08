from scipy.stats import norm
import pandas as pd

NUM_SIMULATIONS = 500

# Define expected customers
mean_customers = 23
sd_customers = 7


def generate_random_numbers(mean, sd):
    """
    Generate random numbers from a normal distribution.
    :param mean: The mean of the distribution.
    :param sd: The standard deviation of the distribution.
    :return:
    """
    random_nums = norm.rvs(loc=mean,
                           scale=sd,
                           size=NUM_SIMULATIONS)
    return random_nums


revenues = generate_random_numbers(mean_customers, sd_customers)


df = pd.DataFrame(columns=['Customers'])

for i in range(0, NUM_SIMULATIONS):
    dictionary = {'Customers': round(revenues[i], 2)}
    df = df._append(dictionary, ignore_index=True)

# Show the data frame which contains results from
# all 500 trials.
print(df)

# Calculate profit summaries.
print("Customers Mean: " + str(df['Customers'].mean()))
print("Customers SD:   " + str(df['Customers'].std()))
print("Customers Min:  " + str(df['Customers'].min()))
print("Customers Max:  " + str(df['Customers'].max()))

# Calculate the risk of incurring a loss.
excessive_customers = df[(df['Customers'] > 30)]
total_days_excessive_customers = excessive_customers['Customers'].count()
chance_of_exceeding = total_days_excessive_customers / NUM_SIMULATIONS
print("Risk of loss: " + str(chance_of_exceeding))
