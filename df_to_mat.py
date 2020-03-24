import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import more_itertools as mit
from itertools import groupby
from tqdm import tqdm
# from collections import OrderedDict

df = pd.read_csv("crimedata-final.csv")

df['time'] = pd.to_datetime(df['occurrencedate']) + pd.to_timedelta(df['occurrencehour'], 'hours')
div = 3.6*(10**12)*2
sub = 1388534400000000000
df['time'] = ((df['time'].astype(int)-sub)/div).astype(int)
df = df.sort_values('time')

# drop_columns = ['occurrencehour', 'occurrencedate', 'Hood_ID', 'ObjectId', 'Division', 'event_unique_id', 'Neighbourhood_x', 'offence', 'premisetype', 'wind_dir', 'unixtime']
# df = df.drop(columns=drop_columns)
# df = pd.get_dummies(df)

time_group = [list(v) for l,v in groupby(df.index, lambda x: df['time'][x])]

df = df[df.columns[[13, 14, 16, 17, 19, 20, 22, 23, 24, 27, 28, 29]]]

# df = df.drop(columns='time')

# xdim = 128
# ydim = 128
adim = len(df.columns)
# # adim = len(df.columns[2::])
# # adim = len(df.columns[3::])
tdim = len(time_group)

# df[['Lat']] = (MinMaxScaler().fit_transform(df[['Lat']])*(xdim-1)).round()
# df[['Long']] = (MinMaxScaler().fit_transform(df[['Long']])*(ydim-1)).round()

# # # matrix = np.zeros((tdim, xdim, ydim, adim))
# # matrix = np.zeros((tdim, xdim, ydim))
# matrix = np.zeros((xdim, ydim, adim))
matrix = np.zeros((tdim, adim))

for i, t in tqdm(enumerate(time_group)):
	time_df = df.iloc[t,:].drop_duplicates()

# c = 0
# for _, row in tqdm(df.iterrows()):
	for x, row in time_df.iterrows():
		matrix[i, :] = np.array(row)
# 	# if (matrix[int(row["Lat"]), int(row["Long"]), :]  == np.zeros((adim))).all():
# 	matrix[int(row["Lat"]), int(row["Long"]), :] = np.array(row[2:])
# 	# else:
# 		# c += 1
# 		# print(c)

# 	# if c == 32*32:
# 		# break
# 		# matrix[i, int(row["Lat"]), int(row["Long"]), :] = np.array(row[3:])
# # 		matrix[i, int(row["Lat"]), int(row["Long"])] += 1

print(matrix[24, :])

np.save('weather', matrix)
print("done!")