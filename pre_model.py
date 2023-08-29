from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import lightgbm as lgb
import pandas as pd
import numpy as np
import cleaned_dataset


# 线性回归模型预测
def get_tv_score(X_train, y_train, X_val, y_val):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    print([y_pred])
    print("Score on Training set: ", lr.score(X_train, y_train))
    print("Score on valid set: ", lr.score(X_val, y_val))
    get_error_table(y_pred, y_val)


def get_error_table(y_pred, y_val):
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_pred, y_val))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_pred, y_val))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_pred, y_val)))
    print('R Squared Error          : ', metrics.r2_score(y_pred, y_val))


# LGBM预测
def lgbm(X_train, y_train, X_val, y_val, X_test):
    params = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'max_depth': 15,
        'metric': 'mse',
        'verbose': -1,
        'seed': 2022,
        'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              eval_metric='rmse')
    y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration_)
    print("model Score on Training set: ", model.score(X_train, y_train))
    print("model Score on valid set: ", model.score(X_val, y_val))
    test = {'price': y_pred_test}
    test = pd.DataFrame(test)
    get_error_table(y_pred, y_val)
    test.loc[:, 'price'] = [c for c in np.exp(test['price'])]
    # print(test)
    # print([np.exp(pre) for pre in y_pred_test])
    return test

