import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
import plotly.express as px


epoch = datetime.date(1970, 1, 1)

def get_day_since_1970(dateString):
    dt = datetime.datetime.strptime(dateString, "%m/%d/%Y").date()
    return (dt - epoch).days

def get_date(days):
    return epoch + datetime.timedelta(days=days)

def rows(filepath):
    for chunk in pd.read_csv(filepath, chunksize=10 ** 6):
        for index, row in chunk.iterrows():
            yield index, row

file = "us-state-data.csv"
mindate = 100000000
maxdate = 0
cases=0
days = pd.DataFrame(data={'count': []}, index=[])

for index, row in tqdm(rows(file), total=26700):
    day = get_day_since_1970(row.submission_date)
    if day not in days.index:
        days.loc[day] = [0]
    maxdate = max(maxdate, day)
    mindate = min(mindate, day)
    days.loc[day, 'count'] += row['new_case']
    cases += row['tot_cases']

days.sort_index(inplace=True)

print(days)

days['total'] = days['count']
print(days.head())

for i,r in days.iterrows():
    days.loc[i, 'date'] = get_date(i)

for i in range(1, days.shape[0]):
    days.iloc[i]['total'] = days.iloc[i-1]['total'] \
                              + days.iloc[i]['count']

for i in range(days.shape[0]-1):
    x = i + 18283 + 1
    days.loc[x, 'total'] = days.loc[x-1, 'total'] \
                         + days.loc[x, 'count']




# 7-day average


for i in range(days.shape[0]-6):
    x = i + 18283 + 6
    days.loc[x, 'sma'] = sum([days.loc[x - n, 'count'] for n in range(7)]) / 7
    days.loc[x, 'diff'] = (days.loc[x, 'count'] - days.loc[x, 'sma'])

print(days)

# 14-day sum


for i in range(days.shape[0]-1):
    x = i + 18283 + 1
    if days.loc[x, 'count'] == 0 or days.loc[x-1, 'count'] == 0:
        continue
    days.loc[x, 'rnought1'] = days.loc[x, 'count'] / days.loc[x-1, 'count']

for i in range(days.shape[0]-14):
    x = i + 18283 + 14
    if sum([days.loc[x - n, 'count'] for n in range(14)]) == 0 or days.loc[x-1, 'count'] == 0:
        continue
    days.loc[x, 'rnought2'] = 14 * days.loc[x, 'count'] / sum([days.loc[x - n, 'count'] for n in range(14)])

for i in range(days.shape[0]-15):
    x = i + 18283 + 15
    if sum([days.loc[x - n, 'count'] for n in range(14)]) == 0:
        continue
    if sum([days.loc[x - n - 1, 'count'] for n in range(14)]) == 0:
        continue
    days.loc[x, 'rnought3'] = sum([days.loc[x - n, 'count'] for n in range(14)]) \
            / sum([days.loc[x - n - 1, 'count'] for n in range(14)])


print(days)


fig = px.scatter(days, x='date', y='rnought1', title="R0 estimate over time (Method 1)",
                 labels={
                     'date':'Date',
                     'rnought1': 'R0'
                 })
fig.update_layout(font=dict(size=30))

fig.show()

fig = px.scatter(days, x='date', y='rnought2', title="R0 estimate over time (Method 2)",
                 labels={
                     'date':'Date',
                     'rnought2': 'R0'
                 })
fig.update_layout(font=dict(size=30))
fig.show()

fig = px.scatter(days, x='date', y='rnought3', title="R0 estimate over time (Method 3)",
                 labels={
                     'date':'Date',
                     'rnought3': 'R0'
                 })
fig.update_layout(font=dict(size=30))
fig.show()


#days.to_csv("daily-increase.csv")