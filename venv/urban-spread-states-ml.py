import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import pearsonr

pinc = pd.read_csv("monthly-state-cases-total.csv").set_index("date")
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
rl1 = indices["POP_RURAL"]
rl2 = indices["POP_ST"] - indices["POP_RURAL"]
rl3 = np.ones((50,1))
rl = pd.concat([rl1, rl2], axis=1)

pinc.drop(["FL", "CA", "TX", "NY"], axis=1, inplace=True)
rl.drop([42, 4, 8, 31], axis=0, inplace=True)
rl1.drop([42, 4, 8, 31], axis=0, inplace=True)
rl2.drop([42, 4, 8, 31], axis=0, inplace=True)
del states[42], states[31], states[8], states[4]


#print(rl.to_numpy().reshape(-1, 1))
#print(np.concatenate((rl.to_numpy(), rl3), axis=1))
rm = rl.to_numpy()





#yuckkk this code is nasty
#print(pinc)

#del pinc['NJ'], density[29], dm[29], states[29], indices['']
#del pinc['AK'], density[1], dm[1], states[1]

#Fl,CA,TX,NY


for i, row in pinc.iterrows():
    #print(row.to_numpy())
    #print(rl)
    #print("hjere")
    #print(row)
    #print(row.to_numpy())
    reg = LinearRegression(copy_X=True).fit(rm, row.to_numpy())
    pinc.loc[row.name, 'reg'] = reg
    pinc.loc[row.name, 'rsq'] = reg.score(rm, row.to_numpy())
    pinc.loc[row.name, 'coef1'], pinc.loc[row.name, 'coef2'] = reg.coef_




print(pinc)



l=['04/01/2020', '09/01/2020']

for test in pinc.index:
    print(test)
    pred = pinc.loc[test, 'reg'].predict(rm)
    resid = (pinc.loc[test].to_numpy()[:46] - pred)
    print(resid)

    fig = px.scatter(x=rl1.to_list(), y=list(resid), text=states, title=test+' pop. density to COVID cases (residuals)',
                     labels={
                         'title':test+' pop. density to COVID cases (residuals)',
                         'x':'Population density',
                         'y':'Residual'
                     })
    fig.update_traces(textposition='top right')

    fig.show()
print('here')