# %%
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


np.random.seed(42)

n = 1000

x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)

rho = 0.9
z = np.random.normal(0, 1, n)
x3 = rho * x2 + np.sqrt(1 - rho**2) * z

error = np.random.normal(0, 3, n)
y = 10 + 2 * x1 - 3 * x2 + error

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'y': y
})

print("Correlation between x2 and x3:", df['x2'].corr(df['x3']))
display(df.head())

# %%

model_2var = smf.ols('y ~ x1 + x3', data=df).fit()
print(model_2var.summary())

model_3var = smf.ols('y ~ x1 + x2 + x3', data=df).fit()
print(model_3var.summary())

# %%
