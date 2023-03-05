import numpy as np
import pandas as pd
df = pd.read_csv('loan.csv')
df.head()
df.drop("Loan_ID", axis=1, inplace=True)
pos=0
neg=0
for i in df['Loan_Status']:
    if i=='Y':
        pos=pos+1
    else:
        neg=neg+1
print(neg,'\n')
print(pos)
