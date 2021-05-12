import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import pearsonr

pinc = pd.read_csv("monthly-state-cases-capita.csv").set_index("date")

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
density = [37.04915775,0.427125397,24.24830135,21.88130572,93.34867336,
           21.44714226,251.3267396,153.7134657,126.6530074,69.68536157,
           51.57125702,8.507445378,85.48777626,71.98736298,21.90390131,
           13.80039887,43.08727641,34.3616568,14.88090535,192.5018829,
           257.2969344,40.2593428,25.35830487,23.62851767,34.12141908,
           2.850101489,9.800494185,10.85432642,56.95420005,411.4245939,
           6.732631138,143.0727546,74.99729538,4.258153658,101.7144826,
           21.89340301,16.64645466,109.0865526,274.472132,61.79339949,
           4.444872803,63.36882175,41.95038683,14.89549849,25.83726813,
           78.11875039,41.78438328,28.60355982,34.76566157,2.280454734]
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

l=['03/01/2021']

for test in l:
    #test = '04/01/2020'
    pred = pinc.loc[test, 'reg'].predict(dm)
    resid = (pinc.loc[test].to_numpy()[:49] - pred)
    print(resid)

    fig = px.scatter(x=density, y=resid, title=test+' pop. density to COVID cases (residuals)',
                     labels={
                         'title':test+' pop. density to COVID cases (residuals)',
                         'x':'Population density',
                         'y':'Residual'
                     })
    fig.update_traces(textposition='top right')
    fig.update_traces(marker_color = '#000', marker_size=18)
    fig.update_layout(title_font=dict(size=36, color="#000"), font=dict(size=42, color="#000"))
    #fig.update_layout(plot_bgcolor="#fff")

    fig.show()
print('here')