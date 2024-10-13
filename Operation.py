import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

data = pd.read_csv('heart_data.csv')
data = data.dropna()
data = data.drop_duplicates()

data['age'] = pd.cut(data['age'], bins=[0, 30, 60, 120], labels=['young', 'middle', 'old'])
data['trestbps'] = pd.cut(data['trestbps'], bins=[0, 110, 140, 200], labels=['low', 'normal', 'high'])
data['chol'] = pd.cut(data['chol'], bins=[0, 120, 240, 1000], labels=['low', 'normal', 'high'])

model = BayesianNetwork([('age','target'),
                         ('sex','target'),
                         ('trestbps','target'),
                         ('chol', 'target')])

model.fit(data, estimator=MaximumLikelihoodEstimator)
print(f"Model is valid? {model.check_model()}")
print()

inference = VariableElimination(model)

age = input("Enter age (young/middle/old):  ").lower()
gender = input("Enter sex (male/female):  ").lower()
sex = 1 if gender == 'male' else 0
trestbps = input("Enter resting blood pressure (low/normal/high):  ").lower()
chol = input("Enter cholestoral level (low/normal/high):  ").lower()
print()

prob = inference.query(variables=['target'],
                       evidence={'age': age, 'sex': sex, 'trestbps': trestbps, 'chol': chol},
                       joint=False)

print(f"Probability of heart attack for {age} {gender} patient with {trestbps} blood pressure and {chol} cholesterol level = {prob['target'].values[0]*100:.2f} %")
print()