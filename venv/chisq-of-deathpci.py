import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
import plotly.express as px


pinc = pd.read_csv("monthly-state-death-total.csv").set_index("date")

chi2, p, dof, exp = chi2_contingency(pinc.to_numpy())
print(chi2)
print(p)
print(dof)
print(exp)

print(pinc.to_numpy() - exp)