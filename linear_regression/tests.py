import wooldridge
import pandas as pd
from ols import OLS

wage = wooldridge.data('wage1')

Y = wage['wage']
X = wage['educ']

model = OLS(Y, X, add_constant=True)
model = model.fit()
model.summary()