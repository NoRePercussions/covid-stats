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

days = pd.read_csv("daily-increase.csv").set_index("Unnamed: 0")
print(days)

for i,r in days.iterrows():
    days.loc[i, 'date'] = get_date(i)

print(r)


fig = px.line(days, x='date', y='total',
                 title="Cumulative US cases by day",
                 labels={
                     'date': 'Date',
                     'total': 'Cumulative cases'
                 },)#- ( x*coef[0] + coef[1] ))
fig.update_layout(font=dict(size=24))
fig.show()