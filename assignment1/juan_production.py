import os
import pandas as pd

def get_loan_prediction(loan_amount_request, current_loan_expenses, dependents, no_of_defaults, coapplicant,
                        property_price, age, income_stability, profession, type_of_employment,
                        expense_type_1, expense_type_2, property_location, credit_score):
    # Create variables for each bin
    income_stability_high = 1 if income_stability == 'High' else 0
    income_stability_low = 1 if income_stability == 'Low' else 0

    profession_pensioner = 1 if profession == 'Pensioners' else 0

    type_of_employment_driver = 1 if type_of_employment == 'Drivers' else 0
    type_of_employment_manager = 1 if type_of_employment == 'Managers' else 0
    type_of_employment_laborer = 1 if type_of_employment == 'Laborers' else 0

    property_location_rural = 1 if property_location == 'Rural' else 0
    property_location_semiurban = 1 if property_location == 'Semi-Urban' else 0
    property_location_urban = 1 if property_location == 'Urban' else 0

    expense_type_1 = 1 if expense_type_1 == 'Y' else 0
    expense_type_2 = 1 if expense_type_2 == 'Y' else 0

    age_65 = 1 if age >= 65 else 0

    credit_score_580_to_620 = 1 if 580 <= credit_score < 620 else 0
    credit_score_620_to_650 = 1 if 620 <= credit_score < 650 else 0
    credit_score_650_to_700 = 1 if 650 <= credit_score < 700 else 0
    credit_score_700_to_750 = 1 if 700 <= credit_score < 750 else 0
    credit_score_750_to_800 = 1 if 750 <= credit_score < 800 else 0
    credit_score_800_to_850 = 1 if 800 <= credit_score < 850 else 0
    credit_score_850_to_900 = 1 if 850 <= credit_score < 900 else 0

    loan_amount_request_0_to_15000 = 1 if 0 <= loan_amount_request < 15000 else 0
    loan_amount_request_15000_to_25000 = 1 if 15000 <= loan_amount_request < 25000 else 0
    loan_amount_request_25000_to_35000 = 1 if 25000 <= loan_amount_request < 35000 else 0
    loan_amount_request_55000_to_80000 = 1 if 55000 <= loan_amount_request < 80000 else 0
    loan_amount_request_80000_to_125000 = 1 if 80000 <= loan_amount_request < 125000 else 0
    loan_amount_request_125000_to_420000 = 1 if 125000 <= loan_amount_request < 420000 else 0

    current_loan_expenses_0_to_150 = 1 if 0 <= current_loan_expenses < 150 else 0
    current_loan_expenses_150_to_250 = 1 if 150 <= current_loan_expenses < 250 else 0
    current_loan_expenses_250_to_350 = 1 if 250 <= current_loan_expenses < 350 else 0
    current_loan_expenses_350_to_450 = 1 if 350 <= current_loan_expenses < 450 else 0
    current_loan_expenses_450_to_550 = 1 if 450 <= current_loan_expenses < 550 else 0
    current_loan_expenses_550_to_700 = 1 if 550 <= current_loan_expenses < 700 else 0
    current_loan_expenses_700_to_1000 = 1 if 700 <= current_loan_expenses < 1000 else 0
    current_loan_expenses_1000_to_1500 = 1 if 1000 <= current_loan_expenses < 1500 else 0


    return - 5676.3417 + 0.6114 * loan_amount_request + 18.6562 * current_loan_expenses - 757.1668 * dependents\
           - 1396.4803 * no_of_defaults + 29203.7244 * coapplicant - 0.0207 * property_price - 3371.0064 * age_65\
           - 8313.1303 * income_stability_high - 6612.5372 * income_stability_low + 12247.441 * profession_pensioner\
           - 1857.9061 * type_of_employment_driver - 1253.6718 * type_of_employment_laborer\
           + 3350.7212 * type_of_employment_manager - 3173.5803 * (not expense_type_1) - 2502.7613 * expense_type_1\
           - 3314.8481 * (not expense_type_2) - 2361.4935 * expense_type_2 - 6176.6418 * property_location_rural\
           - 6036.8106 * property_location_semiurban - 5068.4333 * property_location_urban\
           - 29772.5351 * credit_score_580_to_620 - 27603.4852 * credit_score_620_to_650\
           + 3215.2841 * credit_score_650_to_700 + 3631.6388 * credit_score_700_to_750\
           + 8514.5487 * credit_score_750_to_800 + 9647.3057 * credit_score_800_to_850\
           + 26690.9014 * credit_score_850_to_900 + 2531.6493 * loan_amount_request_0_to_15000\
           + 2667.7943 * loan_amount_request_15000_to_25000 + 2244.7277 * loan_amount_request_25000_to_35000\
           - 2441.9608 * loan_amount_request_55000_to_80000 - 4332.0744 * loan_amount_request_80000_to_125000\
           - 4607.6886 * loan_amount_request_125000_to_420000 - 7948.5741 * current_loan_expenses_0_to_150\
           - 9696.8874 * current_loan_expenses_150_to_250 - 10783.2469 * current_loan_expenses_250_to_350\
           - 12323.2013 * current_loan_expenses_350_to_450 - 15257.2572 * current_loan_expenses_450_to_550\
           - 17568.1113 * current_loan_expenses_550_to_700 - 22517.2451 * current_loan_expenses_700_to_1000\
           - 26794.5916 * current_loan_expenses_1000_to_1500

# Read values from loan_mystery.csv
ROOT_DATA = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
FILE = 'loan_mystery.csv'
df = pd.read_csv(ROOT_DATA + DATASET_DIR + FILE)

# Perform predictions using the model
predictions = []
for index, row in df.iterrows():
   # If any of the values are missing, replace them with 0
    row = row.fillna(0)

    # Perform the prediction for the row and append the result to the predictions list
    prediction = get_loan_prediction(
        loan_amount_request=row['Loan Amount Request (USD)'],
        current_loan_expenses=row['Current Loan Expenses (USD)'],
        dependents=row['Dependents'],
        no_of_defaults=row['No. of Defaults'],
        credit_score=row['Credit Score'],
        coapplicant=row['Co-Applicant'],
        property_price=row['Property Price'],
        age=row['Age'],
        income_stability=row['Income Stability'],
        profession=row['Profession'],
        type_of_employment=row['Type of Employment'],
        expense_type_1=row['Expense Type 1'],
        expense_type_2=row['Expense Type 2'],
        property_location=row['Property Location'])
    predictions.append(prediction)

# Output the predictions to loan_predictions.csv located in the same directory as the loan_mystery.csv file
with open(ROOT_DATA + DATASET_DIR + 'loan_predictions.csv', 'w') as f:
    f.write('Loan Sanction Amount (USD)\n')
    for prediction in predictions:
        f.write(str(prediction) + '\n')
