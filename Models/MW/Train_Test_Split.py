import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data_MW.csv')
X= df[['Entry', 'M', 'R1', 'R2', 'R3', 'T', 'P', 'Al/M', 'Time', 'Cat', 'Cocat']]
Y = df[['MW']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True,random_state=2026)


train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)


train_data = 'train_data_MW.csv'
test_data = 'test_data_MW.csv'


train_df.to_csv(train_data, index=False, encoding='utf-8')
test_df.to_csv(test_data, index=False, encoding='utf-8')


