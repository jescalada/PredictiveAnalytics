import pandas as pd
from sqlalchemy import create_engine
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

# The data file path and file name need to be configured.
CSV_DATA = "retailerDB.csv"
df = pd.read_csv(DATASET_DIR + CSV_DATA)


# Placed query in this function to enable code re-usuability.
def showQueryResult(sql):
    # This code creates an in-memory table called 'Inventory'.
    engine = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name='RetailInventory', con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)

    # This code closes the connection.
    connection.close()
    return queryResult


# Read all rows from the table.
SQL = "SELECT * FROM RetailInventory"
results = showQueryResult(SQL)
print(results)

SQL2 = "SELECT * FROM RetailInventory WHERE price >= 4"
results2 = showQueryResult(SQL2)
print(results2)

# Sort by product name.
SQL3 = "SELECT productID, productName, quantity, price FROM RetailInventory ORDER BY productName"
results3 = showQueryResult(SQL3)
print(results3)

SQL4 = "SELECT productName, vendor, quantity, price FROM RetailInventory ORDER BY productName, quantity"
results4 = showQueryResult(SQL4)
print(results4)

SQL5 = "SELECT DISTINCT productId, productName FROM RetailInventory WHERE vendor NOT IN ('Cadbury', 'Waterford Corp.')"
results5 = showQueryResult(SQL5)
print(results5)

SQL6 = "SELECT productName, price, price * 1.07 AS afterTaxPrice FROM RetailInventory"
results6 = showQueryResult(SQL6)
print(results6)

SQL7 = "SELECT productName, vendor FROM RetailInventory WHERE vendor LIKE '%ry'"
results7 = showQueryResult(SQL7)
print(results7)

SQL8 = "SELECT productName FROM RetailInventory WHERE productName LIKE 'F%'"
results8 = showQueryResult(SQL8)
print(results8)

SQL9 = "SELECT MAX(quantity*price) AS MaxInventoryValue FROM RetailInventory"
results = showQueryResult(SQL9)
print(results.iloc[0]['MaxInventoryValue'])

SQL10 = "SELECT vendor, SUM(price * quantity) AS revenue FROM RetailInventory GROUP BY vendor"
results = showQueryResult(SQL10)
print(results)

SQL11 = "SELECT productName, MIN(price) AS minPrice FROM RetailInventory GROUP BY productName"
results = showQueryResult(SQL11)
print(results)

SQL12 = "SELECT vendor, SUM(quantity) AS TotalItemsStocked FROM RetailInventory \
       GROUP BY vendor HAVING SUM(quantity) > 20"
results = showQueryResult(SQL12)
print(results)

SQL13 = "SELECT vendor, SUM(quantity) AS TotalItemsStocked FROM RetailInventory \
         GROUP BY vendor HAVING SUM(quantity) = 20 OR SUM(quantity) = 34"
results = showQueryResult(SQL13)
print(results)

SQL14 = "SELECT vendor, SUM(price * quantity) AS revenueValue FROM RetailInventory \
         GROUP BY vendor HAVING SUM(price * quantity) > 180"
results = showQueryResult(SQL14)
print(results)