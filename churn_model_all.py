import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.svm import SVC, LinearSVC
from data_clean import DataSelector


data=pd.read_csv('data/churn.csv')
train = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
test = pd.read_csv('data/churn_test.csv',parse_dates=['last_trip_date','signup_date'])

description=train.describe()
info=train.info()

# =============================================================================
data = DataSelector()
X_train,y_train=data.clean_data_gradient_boost(train)
X_test,y_test=data.clean_data_gradient_boost(test)

# =============================================================================
# Histogram matrix, correlation matrix
X_train.hist(figsize=(8,16))
correlation_matrix=X_train.corr()

# =============================================================================
#Grid search
gradient_boosting_grid = {'learning_rate': [0.005, 0.01, 0.02, 0.1],
                        'max_depth': [10],
                      'max_features': [1, 2, 3, 4, 5, 6],
                      'min_samples_leaf': [30, 50, 100],
                      'n_estimators': [100, 1000]}

gdbr_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                             gradient_boosting_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='accuracy')

gdbr_gridsearch.fit(X_train, y_train)
print( "best parameters:", gdbr_gridsearch.best_params_ )
best_gdbr_model = gdbr_gridsearch.best_estimator_
gdbr_best=best_gdbr_model.fit(X_train, y_train)
gdbr_best.score(X_test,y_test)

# =============================================================================
# Gradient Boosting Classifier
gdbr = GradientBoostingClassifier(learning_rate=0.01,
                                  max_depth=10,
                                  n_estimators=1000,
                                  min_samples_leaf=100,
                                  max_features=4)

gdbr.fit(X_train, y_train)
gdbr.score(X_test,y_test) #Achieved accuracy score of 78 pct

# =============================================================================
#Stage score plot
def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    
    mse_train = np.zeros(estimator.n_estimators)
    mse_test = np.zeros(estimator.n_estimators)
    estimator.fit(X_train,y_train)
    for ind, (yh_test,yh_train) in enumerate(zip(estimator.staged_predict(X_test),
                                                 estimator.staged_predict(X_train))):
        mse_test[ind]=np.sum(yh_test!=y_test)/len(y_test)
     
    plt.plot(np.r_[0:estimator.n_estimators],mse_test,
             label ='{} Max features {}'.format(estimator.__class__.__name__, estimator.max_features))
    plt.legend()

gb = GradientBoostingClassifier(learning_rate=0.01,max_depth=10,
                         n_estimators = 1000,min_samples_leaf=100, max_features = 4)
gb2 = GradientBoostingClassifier(learning_rate=0.01,max_depth=10,
                         n_estimators = 1000,min_samples_leaf=100, max_features = 2)
stage_score_plot(gb, X_train, y_train, X_test, y_test)
stage_score_plot(gb2, X_train, y_train, X_test, y_test)

# =============================================================================
#Feature importances
top_cols = np.argsort(gdbr.feature_importances_)
importances =gdbr.feature_importances_[top_cols]
fig = plt.figure(figsize=(10, 10))
x_ind = np.arange(importances.shape[0])
plt.barh(x_ind, importances/importances[-1:], height=.3, align='center')
plt.ylim(x_ind.min() -0.5, x_ind.max() + 0.5)
plt.yticks(x_ind, X_test.columns[top_cols], fontsize=14)
plt.show()

# =============================================================================
#Partial dependence plots
fig, axs = plot_partial_dependence(gdbr, X_train, np.arange(X_train.shape[1]),
                    n_jobs=3, grid_resolution=100,feature_names = X_train.columns)
fig.set_size_inches((20,24))

# =============================================================================
#Other models
# =============================================================================
#SVM
 
#Standardizing/Rescaling continuous variables
cont_vars=['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'surge_pct',
        'avg_surge','trips_in_first_30_days', 'weekday_pct']
X_train[cont_vars]=StandardScaler().fit_transform(X_train[cont_vars])
X_test[cont_vars]=StandardScaler().fit_transform(X_test[cont_vars])
 
svm=SVC(kernel='linear',probability=True)
svm.fit(X_train,y_train)
svm.score(X_test,y_test) #0.695
 
svm=SVC(kernel='poly',probability=True)
svm.fit(X_train,y_train)
svm.score(X_test,y_test) #0.745
 
svm=SVC(kernel='rbf',probability=True)
svm.fit(X_train,y_train)
svm.score(X_test,y_test) #0.7695
 
svm = SVC(gamma='scale') 
svm.fit(X_train,y_train)
svm.score(X_test,y_test) #0.7697
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

# =============================================================================
#Random Forest
rf = RandomForestClassifier(n_estimators=1000,
                            n_jobs=-1)

rf.fit(X_train, y_train)
rf.score(X_test,y_test) #Achieved accuracy score of 75 pct

#Adaptive boosting classifier
abr = AdaBoostClassifier(DecisionTreeClassifier(),
                         learning_rate=0.02,
                         n_estimators=1000)

abr.fit(X_train, y_train)
abr.score(X_test,y_test) #Achieved accuracy score of 73 pct

# =============================================================================
# Logistic regression
# =============================================================================

data = DataSelector()

X_train_logit,y_train_logit=data.clean_data_logit(train)
X_test_logit,y_test_logit=data.clean_data_logit(test)

correlation_matrix=X_train_logit.corr()

#Logistic classifier
#Sklearn
model = LogisticRegression(C=1000)
model.fit(X_train_logit, y_train_logit)
model.coef_

model.score(X_test_logit,y_test_logit) #achieved accuracy of 76.5 pct

#Statsmodels
X_train_logit = sm.add_constant(X_train_logit)
logit = sm.Logit(y_train_logit, X_train_logit)
logit=logit.fit()
logit.summary()














