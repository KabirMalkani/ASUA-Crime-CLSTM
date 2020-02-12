import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("crimedata.csv")

# print(df.head())
# print(df.columns)
print(df[['occurrencedate', 'occurrencehour', 'temperature']][0:20])


xdim = 32
ydim = 32
# tdim = 10
adim = 9

# df = MinMaxScaler().fit_transform(df)
df[['Lat']] = (MinMaxScaler().fit_transform(df[['Lat']])*31).round()
df[['Long']] = (MinMaxScaler().fit_transform(df[['Long']])*31).round()

matrix = np.zeros((xdim, ydim, adim))

location_df = df.iloc[:,[11, 12, *range(40, 49)]].drop_duplicates()

for _, row in location_df.iterrows():
	# print(len(np.array(row[2:])))
	matrix[int(row["Lat"]), int(row["Long"]), :] = np.array(row[2:])
	# print(matrix[int(row["Lat"]), int(row["Long"]), :])


print(matrix[10, 10])
