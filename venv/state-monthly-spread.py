import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

epoch = datetime.date(1970, 1, 1)
epoch = datetime.date(2019, 1, 1)

days = pd.DataFrame(data={'count': []}, index=[])
states = ['AL', 'AK', 'AZ', 'AR', 'CA',
          'CO', 'CT', 'DE', 'FL', 'GA',
          'HI', 'ID', 'IL', 'IN', 'IA',
          'KS', 'KY', 'LA', 'ME', 'MD',
          'MA', 'MI', 'MN', 'MS', 'MO',
          'MT', 'NE', 'NV', 'NH', 'NJ',
          'NM', 'NY', 'NC', 'ND', 'OH',
          'OK', 'OR', 'PA', 'RI', 'SC',
          'SD', 'TN', 'TX', 'UT', 'VT',
          'VA', 'WA', 'WV', 'WI', 'WY']
dates = ['04/01/2020', '05/01/2020', '06/01/2020', '07/01/2020',
         '08/01/2020', '09/01/2020', '10/01/2020', '11/01/2020',
         '12/01/2020', '01/01/2021', '02/01/2021', '03/01/2021', '04/01/2021']
size = [5030053,736081,7158923,3013756,39576757,5782171,3608298,990837,21570527,10725274,1460137,1841377,
                  12822739,6790280,3192406,2940865,4509342,4661468,1363582,6185278,
        7033469,10084442,5709752,2963914,6160281,1085407,1963333,3108462,1379089,9294493,2120220,20215751,10453948,779702,
        11808848,3963516,4241500,13011844,1098163,5124712,887770,6916897,29183290,3275252,643503,8654542,7715946,1795045,5897473,577719]
fancything = pd.Series(data=size, index=states)

print(fancything)

def get_day_since_1970(dateString):
    dt = datetime.datetime.strptime(dateString, "%m/%d/%Y").date()
    return (dt - epoch).days

def get_month_since_2019(dateString):
    dt = datetime.datetime.strptime(dateString, "%m/%d/%Y").date()
    return (dt - epoch).months

def rows(filepath):
    for chunk in pd.read_csv("us-state-data.csv", chunksize=10 ** 6):
        for index, row in chunk.iterrows():
            yield index, row


num_lines = 0
cases = 0

mindate = 10000000
maxdate = 0

monthlydeath = pd.DataFrame([[date] + [0] * 50 for date in dates],
                            columns=['date']+states).set_index('date')
monthlycases = pd.DataFrame([[date] + [0] * 50 for date in dates],
                            columns=['date']+states).set_index('date')

print(monthlydeath)

for index, row in tqdm(rows("us-state-data.csv"), total=26700):
    day = row.submission_date
    if day not in monthlydeath.index or row.state not in states:
        continue
    #print(row.submission_date, row.tot_cases)
    monthlydeath.loc[day, row.state] = row['tot_death']
    monthlycases.loc[day, row.state] = row['tot_cases']

print("cases each month")
print(num_lines)
print(monthlydeath)
print(cases)


pincd = pd.DataFrame([[date]+[0]*50 for date in dates[:-1]],
                       columns=['date']+states).set_index('date')
pincc = pd.DataFrame([[date]+[0]*50 for date in dates[:-1]],
                       columns=['date']+states).set_index('date')

print(pincd)

for i in range(12):
    pincd.iloc[i] = (monthlydeath.iloc[i+1] - monthlydeath.iloc[i])
    pincc.iloc[i] = (monthlycases.iloc[i+1] - monthlycases.iloc[i])

print(pincd)
pincd.to_csv("monthly-state-death-total.csv")
pincc.to_csv("monthly-state-cases-total.csv")

for i in range(12):
    pincd.iloc[i] = pincd.iloc[i] / (fancything)
    pincc.iloc[i] = pincc.iloc[i] / (fancything)

print(pincd)

pincd.to_csv("monthly-state-death-capita.csv")
pincc.to_csv("monthly-state-cases-capita.csv")
