import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from scikeras.wrappers import KerasClassifier


# Function which parameter tunes an elastic net model and 
# returns the best model, best parameters, and the odds associated with the best model coefficients
def tune_logistic(X, y):
    # Initialize model
    lin_model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', class_weight = 'balanced')
    
    # L1 gets from 0 to 100% of penalthy strength incremented by .1
    l1_ratio = [x / 10 for x in range(0, 11)]
    
    # This gets passed to C parameter, which is inverse of regularization strength
    inv_penalty_strength = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    # Define param grid
    params = {
        'C': inv_penalty_strength,
        'l1_ratio': l1_ratio
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(lin_model, 
                               param_grid = params, 
                               cv = StratifiedKFold(n_splits = 5), 
                               scoring = 'roc_auc', 
                               return_train_score = True,
                               n_jobs = 12)
    
    grid_search.fit(X, y)

    grid = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score', ascending = False)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Function to get the odds associated with the best model coefficients
    def get_odds(model, X):
        odds = np.exp(model.coef_).ravel()
        return pd.DataFrame(odds, index = X.columns, columns = ['odds'])
    


    output = {
        'grid': grid,
        'best_score': best_score,
        'best_params': best_params,
        'best_model': best_model,
        'coef_odds': get_odds(best_model, X),
    }
    return output


def monotone_dir(X, y):

    cors = X.apply(lambda x: np.corrcoef(x, y)[0, 1])
    direction = np.where(cors > 0, 1, -1)

    return direction

def tune_xgb(X, y, num_models):

    feature_names = X.columns.to_list()

    mc = monotone_dir(X, y)
    mc = dict(zip(feature_names, mc))
    
    # Calculate the ratio of negative class samples to positive class samples
    class_weight = len(y[y == 0]) / len(y[y == 1])
    
    xgb_to_tune = xgb.XGBClassifier(
        random_state = 123,
        objective = 'binary:logistic',
        scale_pos_weight = class_weight,
        use_label_encoder = False,
        monotone_constraints = mc,
        eval_metric = 'auc'
    )

    max_depth = Integer(1, 14)
    colsample_bytree = Real(0.1, 1)
    subsample = Real(0.3, 1)
    reg_alpha = Real(0, 250)
    reg_lambda = Real(0, 250)
    gamma = Real(0, 5)
    learning_rate = Real(.001, 1, prior = 'log-uniform')
    n_estimators = Integer(100, 1500, prior = 'uniform')

    bayesian_param_grid = dict(
        max_depth = max_depth,
        colsample_bytree = colsample_bytree,
        subsample = subsample,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        gamma = gamma,
        learning_rate = learning_rate,
        n_estimators = n_estimators
    )

    xgb_bayes = BayesSearchCV(
        xgb_to_tune,
        bayesian_param_grid,
        n_iter = num_models,
        cv = StratifiedKFold(n_splits = 10),
        scoring = 'roc_auc',
        n_jobs = 12,
        refit = True,
        return_train_score = True
    )
    xgb_bayes.fit(X, y)

    best_model = xgb_bayes.best_estimator_
    sorted_grid = pd.DataFrame(xgb_bayes.cv_results_).sort_values('mean_test_score', ascending = False)
    best_score = xgb_bayes.best_score_

    output = {
        'grid': sorted_grid,
        'best_score': best_score,
        'best_model': best_model
    }
    return output

def nn(X, activation = 'relu', dropout_rate = 0, HL_neurons = 64, l2_reg = 0, l1_reg = 0, dl1 = 0, num_hidden_layers = 4, lr = 0.01):

    nn = Sequential([
        Dense(HL_neurons, input_shape = (X.shape[1],), activation = activation, kernel_regularizer = L1L2(l1 = l1_reg, l2 = l2_reg)),
        Dropout(dropout_rate)
    ])

    for i in range(1, num_hidden_layers):
        nn.add(BatchNormalization())
        nn.add(Dense(HL_neurons, activation = activation, kernel_regularizer = L1L2(l1 = l1_reg, l2 = l2_reg)))
        nn.add(Dropout(dropout_rate))

    nn.add(Dense(1, activation = 'sigmoid'))
    nn.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr), metrics = ['AUC'])

    return nn

def tune_nn(X, y, num_models, short_run = True):

    batch_size = Integer(5, 1500)
    epochs = Integer(5, 100, prior = 'uniform')
    activation = Categorical(['relu', 'sigmoid'])
    dropout_rate = Real(0, .9, prior = 'uniform')
    num_neurons = Integer(5, 100, prior = 'uniform')
    l2_reg = Real(0.000001, 1, prior = 'log-uniform')
    l1_reg = Real(0.000001, 1, prior = 'log-uniform')
    double_layer1 = Integer(0,1)
    num_hidden_layers = Integer(1,50, prior = 'uniform')

    # short_run to use when developing
    if short_run:
        epochs = Integer(1, 2, prior = 'uniform')
    
    

    bayesian_param_grid = dict(
        batch_size = batch_size,
        epochs = epochs,
        activation = activation,
        dropout_rate = dropout_rate,
        HL_neurons = num_neurons,
        l2_reg = l2_reg,
        l1_reg = l1_reg,
        dl1 = double_layer1,
        num_hidden_layers = num_hidden_layers,
        lr = Real(0.0001, 0.1, prior = 'log-uniform')
    )


    # To resolve this error:
    # TypeError: Cannot clone object '<Sequential name=sequential_7, built=True>' (type <class 'keras.src.models.sequential.Sequential'>): it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method.

    bayesian_grid_estimator = KerasClassifier(build_fn = nn, verbose = 0)
    
    bayesian_grid = BayesSearchCV(
        bayesian_grid_estimator,
        bayesian_param_grid,
        n_iter = num_models,
        cv = StratifiedKFold(n_splits = 5),
        scoring = 'roc_auc',
        n_jobs = 12,
        refit = True,
        return_train_score = True
    )

    grid = bayesian_grid.fit(X, y)

    return grid
