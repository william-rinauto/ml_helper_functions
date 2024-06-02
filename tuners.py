import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

from functools import partial

from scipy.stats import randint, uniform



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

def tune_xgb_bayes(X, y, X_val, y_val, num_models):

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
        eval_metric = 'auc',
        n_estimators = 1000
    )

    max_depth = Integer(1, 14)
    colsample_bytree = Real(0.1, 1)
    subsample = Real(0.3, 1)
    
    # L1 regularization term on weights
    reg_alpha = Real(0, 250)
    
    # L2 regularization term on weights
    reg_lambda = Real(0, 250)
    
    # Minimum loss reduction required to make a further partition on a leaf node of the tree
    gamma = Real(0, 5)
    learning_rate = Real(.001, 1, prior = 'log-uniform')

    bayesian_param_grid = dict(
        max_depth = max_depth,
        colsample_bytree = colsample_bytree,
        subsample = subsample,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        gamma = gamma,
        learning_rate = learning_rate,
    )

    xgb_bayes = BayesSearchCV(
        xgb_to_tune,
        bayesian_param_grid,
        n_iter = num_models,
        cv = StratifiedKFold(n_splits = 10),
        scoring = 'roc_auc',
        n_jobs = 12,
        refit = True,
        return_train_score = True,
    )
    xgb_bayes.fit(X, y, eval_set = [(X_val, y_val)], early_stopping_rounds = 10, verbose = 0)

    best_model = xgb_bayes.best_estimator_
    sorted_grid = pd.DataFrame(xgb_bayes.cv_results_).sort_values('mean_test_score', ascending = False)
    best_score = xgb_bayes.best_score_

    output = {
        'grid': sorted_grid,
        'best_score': best_score,
        'best_model': best_model
    }
    return output

def tune_xgb_random(X, y, X_val, y_val, num_models):
    feature_names = X.columns.to_list()

    mc = monotone_dir(X, y)
    mc = dict(zip(feature_names, mc))
    
    # Calculate the ratio of negative class samples to positive class samples
    class_weight = len(y[y == 0]) / len(y[y == 1])
    
    xgb_to_tune = xgb.XGBClassifier(
        random_state=123,
        objective='binary:logistic',
        scale_pos_weight=class_weight,
        use_label_encoder=False,
        monotone_constraints=mc,
        eval_metric='auc',
        n_estimators=1000
    )

    param_dist = {
        'max_depth': randint(1, 14),
        'colsample_bytree': uniform(0.1, 0.9),
        'subsample': uniform(0.3, 0.7),
        'reg_alpha': uniform(0, 250),
        'reg_lambda': uniform(0, 250),
        'gamma': uniform(0, 5),
        'learning_rate': uniform(0.001, 0.999),
    }

    xgb_random = RandomizedSearchCV(
        xgb_to_tune,
        param_distributions=param_dist,
        n_iter=num_models,
        cv=StratifiedKFold(n_splits=10),
        scoring='roc_auc',
        n_jobs=12,
        refit=True,
        return_train_score=True
    )

    xgb_random.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=0)

    best_model = xgb_random.best_estimator_
    sorted_grid = pd.DataFrame(xgb_random.cv_results_).sort_values('mean_test_score', ascending=False)
    best_score = xgb_random.best_score_

    output = {
        'grid': sorted_grid,
        'best_score': best_score,
        'best_model': best_model
    }

    return output

def nn(input_dim, activation = 'relu', dropout_rate = 0, HL_neurons = 64, l2_reg = 0, l1_reg = 0, dl1 = 0, num_hidden_layers = 4, lr = 0.01):

    nn = Sequential()
    nn.add(Input(shape=(input_dim,)))
    nn.add(Dense(HL_neurons, activation=activation, kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg)))
    nn.add(Dropout(dropout_rate))

    for i in range(1, num_hidden_layers):
        nn.add(BatchNormalization())
        nn.add(Dense(HL_neurons, activation = activation, kernel_regularizer = L1L2(l1 = l1_reg, l2 = l2_reg)))
        nn.add(Dropout(dropout_rate))

    nn.add(Dense(1, activation = 'sigmoid'))
    nn.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr), metrics = ['AUC'])

    return nn

def tune_nn_bayes(X, y, X_val, y_val, num_models, short_run = True):

    batch_size = Integer(5, 1500)
    activation = Categorical(['relu', 'sigmoid'])
    dropout_rate = Real(0, .9, prior = 'uniform')
    num_neurons = Integer(5, 100, prior = 'uniform')
    l2_reg = Real(0.000001, 1, prior = 'log-uniform')
    l1_reg = Real(0.000001, 1, prior = 'log-uniform')
    double_layer1 = Integer(0,1)
    num_hidden_layers = Integer(1,5, prior = 'uniform')

    # short_run to use when developing
    if short_run:
        epochs = Integer(1, 2, prior = 'uniform')
    
    

    bayesian_param_grid = dict(
        batch_size = batch_size,
        model__activation = activation,
        model__dropout_rate = dropout_rate,
        model__HL_neurons = num_neurons,
        model__l2_reg = l2_reg,
        model__l1_reg = l1_reg,
        model__dl1 = double_layer1,
        model__num_hidden_layers = num_hidden_layers,
        model__lr = Real(0.0001, 0.1, prior = 'log-uniform')
    )


    # To resolve this error:
    # TypeError: Cannot clone object '<Sequential name=sequential_7, built=True>' (type <class 'keras.src.models.sequential.Sequential'>): it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method.

    input_dim = X.shape[1]
    # bayesian_grid_estimator = KerasClassifier(model=nn, input_dim=input_dim, verbose = 0)
    # bayesian_grid_estimator._estimator_type = 'classifier'
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    bayesian_grid_estimator = KerasClassifier(model=nn, input_dim=input_dim, verbose = 0, callbacks=[early_stopping])
    

    bayesian_grid = BayesSearchCV(
        bayesian_grid_estimator,
        bayesian_param_grid,
        n_iter = num_models,
        cv = StratifiedKFold(n_splits = 5),
        scoring = 'roc_auc',
        n_jobs = -1,
        refit = True,
        return_train_score = True
    )

    bayesian_grid.fit(X, y, validation_data=(X_val, y_val))

    best_model = bayesian_grid.best_estimator_
    best_score = bayesian_grid.best_score_
    sorted_grid = pd.DataFrame(bayesian_grid.cv_results_).sort_values('mean_test_score', ascending = False)

    output = {
        'grid': sorted_grid,
        'best_score': best_score,
        'best_model': best_model
    }

    return output

def tune_nn_random(X, y, X_val, y_val, num_models):
    input_dim = X.shape[1]
    
    param_dist = {
        'batch_size': randint(5, 1500),
        'model__activation': ['relu', 'sigmoid'],
        'model__dropout_rate': uniform(0, 0.9),
        'model__HL_neurons': randint(5, 100),
        'model__l2_reg': uniform(0.000001, 1),
        'model__l1_reg': uniform(0.000001, 1),
        'model__dl1': randint(0, 1),
        'model__num_hidden_layers': randint(1, 5),
        'model__lr': uniform(0.0001, 0.1)
    }
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model = KerasClassifier(model=nn, input_dim=input_dim, verbose=0, callbacks=[early_stopping])
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=num_models,
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    
    random_search.fit(X, y, validation_data=(X_val, y_val))
    
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_
    sorted_grid = pd.DataFrame(random_search.cv_results_).sort_values('mean_test_score', ascending=False)
    
    output = {
        'grid': sorted_grid,
        'best_score': best_score,
        'best_model': best_model
    }
    
    return output


def rebuild_best_nn(X, y, X_val, y_val, best_model):

    model_params = best_model.get_params()

    batch_size = model_params['batch_size']
    activation = model_params['model__activation']
    dropout_rate = model_params['model__dropout_rate']
    num_neurons = model_params['model__HL_neurons']
    l2_reg = model_params['model__l2_reg']
    l1_reg = model_params['model__l1_reg']
    double_layer1 = model_params['model__dl1']
    num_hidden_layers = model_params['model__num_hidden_layers']
    lr = model_params['model__lr']

    input_dim = X.shape[1]

    model = nn(input_dim, activation, dropout_rate, num_neurons, l2_reg, l1_reg, double_layer1, num_hidden_layers, lr)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X, y, validation_data=(X_val, y_val), batch_size=batch_size, epochs=100, verbose=1, callbacks=[early_stopping])
        
    return history
