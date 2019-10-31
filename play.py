import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_csv("FamilyIncome.csv", comment="#")

result = smf.ols("expenditure ~ income", data=df).fit()
b0, b1 = result.params

new_data = {"income": [1100, 1200]}
df = pd.DataFrame(new_data)
pred = result.predict(df)

print(pred)