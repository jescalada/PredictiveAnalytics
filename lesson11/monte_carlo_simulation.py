from scipy.stats import norm
import pandas as pd

NUM_SIMULATIONS = 500

# Define revenue, fixed and variable cost parameters.
REV_EXPECTED = 194000
REV_SD = 15000

FC_EXPECTED = 60000
FC_SD = 4000

VC_EXPECTED = 100000
VC_SD = 40000


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


revenues = generate_random_numbers(REV_EXPECTED, REV_SD)
fixedCosts = generate_random_numbers(FC_EXPECTED, FC_SD)
variableCosts = generate_random_numbers(VC_EXPECTED, VC_SD)

profits = []

df = pd.DataFrame(columns=['Revenue', 'Fixed Cost',
                           'Variable Cost', 'Profit'])

for i in range(0, NUM_SIMULATIONS):
    profit = revenues[i] - fixedCosts[i] - variableCosts[i]

    dictionary = {'Revenue': round(revenues[i], 2),
                  'Fixed Cost': round(fixedCosts[i], 2),
                  'Variable Cost': round(variableCosts[i], 2),
                  'Profit': round(profit, 2)}
    df = df._append(dictionary, ignore_index=True)

# Show the data frame which contains results from
# all 500 trials.
print(df)

# Calculate profit summaries.
print("Profit Mean: " + str(df['Profit'].mean()))
print("Profit SD:   " + str(df['Profit'].std()))
print("Profit Min:  " + str(df['Profit'].min()))
print("Profit Max:  " + str(df['Profit'].max()))

# Calculate the risk of incurring a loss.
dfLoss = df[(df['Profit'] < 0)]
totalLosses = dfLoss['Profit'].count()
riskOfLoss = totalLosses / NUM_SIMULATIONS
print("Risk of loss: " + str(riskOfLoss))
