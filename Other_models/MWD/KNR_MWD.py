import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df_train = pd.read_csv('train_data_MWD.csv')
df_test = pd.read_csv('test_data_MWD.csv')
x_train = df_train[['M', 'R1', 'R2', 'R3', 'T', 'P', 'Al/M', 'Time', 'Cat', 'Cocat']]
y_train = df_train['MWD']
x_test = df_test[['M', 'R1', 'R2', 'R3', 'T', 'P', 'Al/M', 'Time', 'Cat', 'Cocat']]
y_test = df_test['MWD']




def bayesian_optimize(n_neighbors):
    clf = KNeighborsRegressor(n_neighbors=int(n_neighbors))
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    scores = cross_val_score(clf, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    return scores.mean()



pbounds = {
    'n_neighbors': (1, 20)
}


optimizer = BayesianOptimization(
    f=bayesian_optimize,
    pbounds=pbounds,
    random_state=24,
)


optimizer.maximize(init_points=10, n_iter=60)


best_params = optimizer.max['params']
print("Best parameters found: ", best_params)

clf_best = KNeighborsRegressor(n_neighbors=int(best_params['n_neighbors']))


clf_best.fit(x_train, y_train)




y_train_pred = clf_best.predict(x_train)
y_test_pred = clf_best.predict(x_test)



train_score_mse = mean_squared_error(y_train,y_train_pred)
test_score_mse = mean_squared_error(y_test,y_test_pred)
train_score_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_score_rmse  =  np.sqrt(mean_squared_error(y_test,y_test_pred))
train_score_mae = mean_absolute_error(y_train,y_train_pred)
test_score_mae = mean_absolute_error(y_test,y_test_pred)
train_r2_score = r2_score(y_train,y_train_pred)
test_r2_score = r2_score(y_test,y_test_pred)

print('MSE',train_score_mse,test_score_mse)
print('RMSE',train_score_rmse,test_score_rmse)
print('MAE',train_score_mae,test_score_mae)
print('R2',train_r2_score,test_r2_score)



# import joblib
# joblib.dump(clf_best, './KNR_MWD.pkl')
#
#

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


font={'family':"Times New Roman",'size':'26'}
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

plt.rc('font',**font)
plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(figsize = (10, 10))

ax.set_aspect('equal')
ax.scatter(y_train, y_train_pred, label="Training Set",alpha=1, marker='o', s=42, facecolor='#1f77b4', color='#1f77b4')
ax.scatter(y_test, y_test_pred, label='Test Set',alpha=1, marker='o', s=42, facecolor='#ff7f0e', color='#ff7f0e')
ax.tick_params(axis='both', labelcolor='black', width=3, length=3, color='black', which='major')
y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
x_major_locator=MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(xmin= 0,xmax = 7)
ax.set_ylim(ymin= 0,ymax = 7)
padding = 0.05 * (7 - 0)

ax.set_xlim(xmin=0 - padding, xmax=7 + padding)
ax.set_ylim(ymin=0 - padding, ymax=7 + padding)

line= ax.plot([0 - padding, 7 + padding], [0 - padding, 7 + padding], transform=ax.transAxes, ls='--', c='gray', alpha=0.8)


plt.setp(line, linewidth=1)

plt.title('MWD--KNR')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('True MWD ',fontsize=18)
plt.ylabel('Predicted MWD ',fontsize=18)

plt.legend(markerscale=2,frameon=True,fontsize=18,labelspacing=0.8,handlelength=3)
r2_score= format(test_r2_score, '.4f')
score_mae= format(test_score_mae, '.4f')
plt.text(5.5,0.5,f'R$^2$ : {r2_score} \nMAE : {score_mae} ',fontsize=20,linespacing=2)
# plt.savefig('MWD_KNR.svg',dpi=600, bbox_inches = "tight")
plt.show()




