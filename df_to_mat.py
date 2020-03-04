import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import more_itertools as mit
from itertools import groupby
from tqdm import tqdm

df = pd.read_csv("crimedata.csv")

df['time'] = pd.to_datetime(df['occurrencedate']) + pd.to_timedelta(df['occurrencehour'], 'hours')
drop_columns = ['occurrencehour', 'occurrencedate', 'Hood_ID', 'ObjectId', 'Division', 'event_unique_id', 'Neighbourhood_x', 'offence', 'premisetype', 'wind_dir', 'unixtime']
df = df.drop(columns=drop_columns)
df = pd.get_dummies(df)

div = (10**9)*3600*6
time_group = (df['time'].astype(int)/div).astype(int)
time_group = [list(v) for l,v in groupby(df.index, lambda x: time_group[x])]

df = df.drop(columns='time')

xdim = 16
ydim = 16
adim = len(df.columns[3::])
tdim = len(time_group)

df[['Lat']] = (MinMaxScaler().fit_transform(df[['Lat']])*(xdim-1)).round()
df[['Long']] = (MinMaxScaler().fit_transform(df[['Long']])*(ydim-1)).round()

matrix = np.zeros((tdim, xdim, ydim, adim))

for i, t in tqdm(enumerate(time_group)):
	time_df = df.iloc[t,:].drop_duplicates()

	for _, row in time_df.iterrows():
		matrix[i, int(row["Lat"]), int(row["Long"]), :] = np.array(row[3:])

np.save('matrix', matrix)
