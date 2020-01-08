import pandas as pd

df = pd.read_csv(r"MCI_2014_to_2018.csv")

df = df.groupby(["Neighbourhood", "offence"]).size().reset_index()
df.columns = ["area", "crime", "count"]
totals = df.groupby("area")["count"].sum()

df = df.pivot(index='area', columns='crime', values='count')
df["Total"] = totals

print(df)
