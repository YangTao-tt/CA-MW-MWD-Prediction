import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_CA.csv')
X= df[['Entry', 'M', 'R1', 'R2', 'R3', 'T', 'P', 'Al/M', 'Time', 'Cat', 'Cocat']]
Y = df[['CA']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True,random_state=2021)

train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)

train_data_activity = 'train_data_CA.csv'
test_data_activity = 'test_data_CA.csv'

train_df.to_csv(train_data_activity, index=False, encoding='utf-8')
test_df.to_csv(test_data_activity, index=False, encoding='utf-8')