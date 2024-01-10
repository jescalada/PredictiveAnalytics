import pandas as pd

co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]
df = pd.DataFrame({'CO2': co2}, index=pd.date_range('09-01-2022',
                                                    periods=len(co2), freq='B'))
df['CO2_t-1'] = df['CO2'].shift(periods=1)
df['CO2_t-2'] = df['CO2'].shift(periods=2)
df.dropna(inplace=True)

print(df)
