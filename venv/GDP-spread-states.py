import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import pearsonr

pinc = pd.read_csv("monthly-state-cases-total.csv").set_index("date")

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
density = [23054, 55759, 372522, 135225, 3183251,
           396367, 288985, 76410, 1111378, 625329,
           98536, 82265, 908913, 381733, 197172,
           175703, 217564, 267051, 68441, 434312,
           604208, 548567, 385907, 120429, 336816,
           52948, 129098, 180406, 89836, 652412,
           105263, 1751674, 596383, 57400, 706764,
           207381, 255418, 824603, 64441, 249958,
           54057, 385741, 1918065, 192013, 35271,
           561846, 610488, 78507, 351922, 39794]
size = [3608298,990837,21570527,10725274,1460137,1841377,12822739,6790280,3192406,2940865,4509342,4661468,1363582,6185278,
        7033469,10084442,5709752,2963914,6160281,1085407,1963333,3108462,1379089,9294493,2120220,20215751,10453948,779702,
        11808848,3963516,4241500,13011844,1098163,5124712,887770,6916897,29183290,3275252,643503,8654542,7715946,1795045,5897473,577719]
dm = [[item] for item in density]



print(pinc)

del pinc['NJ'], density[29], dm[29], states[29]
#del pinc['AK'], density[1], dm[1], states[1]

for i, row in pinc.iterrows():
    reg = LinearRegression().fit(dm, row.to_numpy())
    ols = sm.OLS(row.to_numpy(), sm.add_constant(dm))
    results = ols.fit()
    pinc.loc[row.name, 'reg'] = reg
    pinc.loc[row.name, 'rsq'] = reg.score(dm, row.to_numpy())

    t = results.tvalues
    p = results.pvalues
    #print(t)
    #print(p)
    #print(results.params)
    #print(reg.coef_)
    #print(results.rsquared)
    #print(reg.score(dm, row.to_numpy()))

    #pinc.loc[row.name, 't'] = t[1]
    #pinc.loc[row.name, 'p'] = p[1]
    rm = pearsonr(density, row.to_numpy())
    pinc.loc[row.name, 'r'] = rm[0]
    pinc.loc[row.name, 'p'] = rm[1]
    #print(00/0)


print(pinc)

l=['04/01/2020', '05/01/2020', '09/01/2020',
   '10/01/2020', '11/01/2020', '01/01/2021',
   '02/01/2021', '03/01/2021',]

for test in l:
    #test = '04/01/2020'
    pred = pinc.loc[test, 'reg'].predict(dm)
    resid = (pinc.loc[test].to_numpy()[:49] - pred)
    print(resid)

    fig = px.scatter(x=density, y=resid, text=states, title='April 2020 pop. density to COVID cases (residuals)',
                     labels={
                         'title':'April 2020 pop. density to COVID cases (residuals)',
                         'x':'Population density',
                         'y':'Residual'
                     })
    fig.update_traces(textposition='top right')

    fig.show()
print('here')