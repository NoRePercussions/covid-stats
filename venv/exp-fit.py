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

days = pd.read_csv("daily-increase.csv").set_index("Unnamed: 0")
print(days)

y = np.array([])
const = min(days.loc[18523]) - 1
for i in range(18525, 18571):
    y = np.append(y, days.loc[i, 'count'] - const)

print(y)
x = np.array(list(range(len(y))))


coef  = np.polyfit(x, np.log(y), 1)
print(coef)

fig = px.scatter(x=[get_date(int(a+18523)) for a in x],  y=y+const,
                 title="Cumulative cases by day from 9/18/20 to 12/5/20",
                 labels={
                     'x': 'Date',
                     'y': 'Number of cumulative cases'
                 })#- ( x*coef[0] + coef[1] ))
fig.show()

fig = px.scatter(x=[get_date(int(a+18523)) for a in x],  y=np.log(y),
                 title="Transformed cases by day from 9/18/20 to 12/5/20",
                 labels={
                     'x': 'Date',
                     'y': 'LN of translated cumulative cases'
                 })#- ( x*coef[0] + coef[1] ))
fig.show()

fig = px.scatter(x=[get_date(int(a+18523)) for a in x],  y=np.log(y)- ( x*coef[0] + coef[1] ),
                 title="Residuals of exponential fit of covid cases (r^2 = 0.80)",
                 labels={
                     'x': 'Date',
                     'y': 'Residual'
                 })#)
fig.update_traces(marker_color = '#000', marker_size=18)
fig.update_layout(font=dict(size=48, color="#000"))

fig.show()

correlation_matrix = np.corrcoef(x, np.log(y))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print(r_squared)

#18523 to 18601
