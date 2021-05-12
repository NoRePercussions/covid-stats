import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

epoch = datetime.date(1970, 1, 1)
days = pd.DataFrame(data={'count': []}, index=[])
states = ['AL', 'AK', 'AZ', 'AR', 'CA',
          'CO', 'CT', 'DE', 'FL', 'GA',
          'HI', 'ID', 'I;', 'IN', 'IA',
          'KS', 'KY', 'LA', 'ME', 'MD',
          'MA', 'MI', 'MN', 'MS', 'MO',
          'MT', 'NE', 'NV', 'NH', 'NJ',
          'NM', 'NY', 'NC', 'ND', 'OH',
          'OK', 'OR', 'PA', 'RI', 'SC',
          'SD', 'TN', 'TX', 'UT', 'VT',
          'VA', 'WA', 'WV', 'WI', 'WY']


def get_day_since_1970(dateString):
    dt = datetime.datetime.strptime(dateString, "%m/%d/%Y").date()
    return (dt - epoch).days

def rows(filepath):
    for chunk in pd.read_csv(filepath, chunksize=10 ** 6):
        for index, row in chunk.iterrows():
            yield index, row


num_lines = 0
cases = 0

mindate = 10000000
maxdate = 0

for index, row in tqdm(rows("us-case-data.csv")):
    continue
    day = get_day_since_1970(row.submission_date)
    if day not in days.index:
        days.loc[day] = [0]
        maxdate = max(maxdate, day)
        mindate = min(mindate, day)
    days.loc[day, 'count'] += row['tot_cases']
    cases += row['tot_cases']

print(num_lines)
print(days.head())
print(cases)
