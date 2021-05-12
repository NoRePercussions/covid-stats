import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import pearsonr

pinc = pd.read_csv("monthly-state-cases-capita.csv").set_index("date")
indices = pd.read_csv("PctUrbanRural_State.csv")

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

indices['acro'] = states
indices.set_index('acro')

important = "POPPCT_RURAL"
#print(indices)
rl = indices["POPPCT_RURAL"]
#print(rl.to_numpy().reshape(-1, 1))
rm = rl.to_numpy().reshape(-1, 1).tolist()





#yuckkk this code is nasty
#print(pinc)

#del pinc['NJ'], density[29], dm[29], states[29], indices['']
#del pinc['AK'], density[1], dm[1], states[1]

for i, row in pinc.iterrows():
    #print(row.to_numpy())
    #print(rl)
    #print("hjere")
    #print(rm)
    #print(row)
    #print(row.to_numpy())
    rm = rl.to_numpy().reshape(-1, 1).tolist()
    reg = LinearRegression(copy_X=True).fit(rm, row.to_numpy())
    ols = sm.OLS(row.to_numpy(), sm.add_constant(rl))
    results = ols.fit()
    pinc.loc[row.name, 'reg'] = reg
    pinc.loc[row.name, 'rsq'] = reg.score(rm, row.to_numpy())

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
    rm = pearsonr(rl, row.to_numpy())
    pinc.loc[row.name, 'r'] = rm[0]
    pinc.loc[row.name, 'p'] = rm[1]
    #print(00/0)


#print(pinc)

l=['05/01/2020']

stop

for test in l:
    #test = '04/01/2020'
    rm = rl.to_numpy().reshape(-1, 1).tolist()
    pred = pinc.loc[test, 'reg'].predict(rm)
    resid = (pinc.loc[test].to_numpy()[:50] - pred)
    print(resid)

    fig = px.scatter(x=rl.to_list(), y=list(resid), text=states, title='April 2020 pop. density to COVID cases (residuals)',
                     labels={
                         'title':'April 2020 pop. density to COVID cases (residuals)',
                         'x':'Population density',
                         'y':'Residual'
                     })
    fig.update_traces(textposition='top right')

    fig.show()
print('here')