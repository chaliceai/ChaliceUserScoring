#Installing all necessary packages 

import numpy as np
from dateutil.easter import *
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

def testtrainsplit(X,Y,test_proportion):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_proportion, random_state=0)
    return X_train, X_val, Y_train, Y_val


def run_smote(X_train, Y_train, sampling, knn):
    
    X_smote = X_train
    Y_smote = Y_train

    categorical_features_indices = np.where(X_smote.dtypes != np.float64)[0]
    print(categorical_features_indices)
    sm = SMOTENC(categorical_features= categorical_features_indices, random_state=0, sampling_strategy=sampling, k_neighbors=knn)
    X_train_smote, Y_train_smote = sm.fit_resample(X_smote, Y_smote)
    return X_train_smote, Y_train_smote

def catmodel(X_train, X_val, Y_train, Y_val, bootstrap_type, class_weights, depth, learning_rate,loss_function):
    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    print(categorical_features_indices)

    clf = CatBoostClassifier(
        iterations=200,
        bootstrap_type = bootstrap_type,
        random_strength = 0.5,
        class_weights = class_weights,
        custom_metric=['F1', 'AUC', 'Accuracy', 'Recall', 'Precision'],
        depth = depth,
        l2_leaf_reg = 5,
        learning_rate=learning_rate, 
        loss_function=loss_function
    )

    clf.fit(X_train, Y_train, 
            cat_features=categorical_features_indices, 
            eval_set=(X_val, Y_val),
            use_best_model=True,
            plot = True, 
            verbose=False)
    print('CatBoost model is fitted: ' + str(clf.is_fitted()))
    print('CatBoost model parameters:' + str(clf.get_params()))
    print('Best Score:' + str(clf.get_best_score()))
    print('Best Iteration:' + str(clf.get_best_iteration()))
    
    import shap
    shap_values = clf.get_feature_importance(Pool(X_val, label=Y_val,cat_features=categorical_features_indices), 
                                                                     type="ShapValues")
    expected_value = shap_values[0,-1]
    shap_values = shap_values[:,:-1]

    shap.initjs()
    shap.force_plot(expected_value, shap_values[3,:], X_val.iloc[3,:])
    shap.summary_plot(shap_values, X_val, show = False)
    plt.savefig('shap_plot.png')
    
    return clf, categorical_features_indices
    
    
def catboost_smote(X_train, Y_train, X_val, Y_val, use_smote):
    
    if use_smote == True:
        sampling = float(input('Enter Sampling Proportion:'))
        knn = int(input('Enter value for knn:'))
        run_smote(X_train, Y_train, sampling, knn)
        clf, categorical_feature_indices = catmodel(X_train_smote, X_val, Y_train_smote, Y_val)
        
    else: clf, categorical_feature_indices = catmodel(X_train, X_val, Y_train, Y_val)
        
    return clf, categorical_feature_indices
        
def reduce_model_dim(drop_columns, X_train, X_val):
    X_train = X_train.drop(drop_columns, 1)
    X_val = X_val.drop(drop_columns, 1)
    return X_train, X_val
        
def grid_search(cat_cols, X, Y):
    #Hyperparameter tuning can be used to improve the model. 
    #This takes a long time to run 
    #Making scoring function 
    from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, recall_score, f1_score

    def custom_scorer(y_true, y_pred, actual_scorer):
        score = np.nan
    
        try:
          score = actual_scorer(y_true, y_pred)
        except Exception: 
          pass

        return score
    #Set up metrics for Grid Search
    acc = make_scorer(custom_scorer, actual_scorer = accuracy_score)
    auc_score = make_scorer(custom_scorer, actual_scorer = roc_auc_score, 
                            needs_threshold=True) # <== Added this to get correct roc
    recall = make_scorer(custom_scorer, actual_scorer = recall_score)
    f1 = make_scorer(custom_scorer, actual_scorer = f1_score)
    #Grid Search Exploration
    from sklearn.model_selection import GridSearchCV
    # convert categorical columns to integers
    #Format is ['col1', 'col2', 'col3']
    category_cols = cat_cols
    for header in category_cols:
        X[header] = X[header].astype('category').cat.codes

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = CatBoostClassifier()
    parameters = {'depth'         : [6,8,10],
                'learning_rate' : [0.05, 0.1,0.5],
                'iterations'    : [100,250,300],
                'bootstrap_type' : ["MVS", "Bayesian", "Bernoulli"],
                'loss_function' : ['Logloss'],
                'class_weights' : [[0.3,0.7], [0.4, 0.6], [0.5,0.5], [0.1,0.9]]
                     }
                    
    grid = GridSearchCV(estimator=clf, 
                        param_grid = parameters, 
                        cv = 3, 
                        scoring = {"roc_auc": auc_score, "accuracy": acc, "recall": recall},
                        refit="recall",
                        return_train_score = True
                       )

    grid.fit(X_train, Y_train)
    print(grid.cv_results_)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
    print("\n The best score across ALL searched params:\n", grid.best_score_)
    print("\n The best parameters across ALL searched params:\n", grid.best_params_)
    
def cross_validation(iterations, learning_rate, X, Y, cat_feature_indices, stratified_bool_val):
    ###Cross Validation for Model Assessment
    from catboost import cv

    params = {}
    params['loss_function'] = 'Logloss'
    params['iterations'] = iterations
    params['custom_loss'] = ['Accuracy', 'Precision', 'Recall', 'F1']
    params['random_seed'] = 63
    params['learning_rate'] = learning_rate

    cv_data = cv(
        params = params,
        pool = Pool(X, label=Y, cat_features=cat_feature_indices),
        fold_count=5,
        shuffle=True,
        partition_random_seed=0,
        plot=True,
        stratified=stratified_bool_val,
        verbose=False)
    return cv_data