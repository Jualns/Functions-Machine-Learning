import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

def cv_estimate(cv_clf, params, X, y, kf):
    '''Faz os modelos do tipo AdaBoostClassfier e GradientBoostingClassifier com kf para várias árvores de forma eficiente'''
    val_scores_bal = np.zeros((params['n_estimators'],), dtype=np.float64)
    for train, test in kf.split(X, y):
        cv_clf.fit(X.iloc[train], y.iloc[train])
        val_scores_bal += heldout_score(cv_clf, X.iloc[test], y.iloc[test])
    val_scores_bal /= kf.get_n_splits()
    return val_scores_bal

def possibilidades(vet_a, tam_vet):
    vet = list(range(1,len(vet_a)))
    aux = pd.DataFrame(list(product(vet, repeat=tam_vet)))
    cols = list(aux.columns)
    aux.sort_values(by=cols).drop_duplicates()
    aux = aux.sort_values(by=cols).drop_duplicates()
    return aux[aux.apply(lambda x: sum(x<=vet_a) == tam_vet, axis = 1)].reset_index(drop = True)

def heldout_score(clf, X_test, y_test):
    '''Gera o score'''
    
    score_balanced = np.zeros((clf.n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        score_balanced[i] = balanced_accuracy_score(y_test, y_pred)
    return score_balanced



def select_boosting(multi_params, kf, X_train, y_train):
    all_params = list(multi_params.keys())
    num_var = len(all_params)
    number = np.ones(num_var, int)
    for param in range(num_var):
        try:
            number[param] = len(multi_params[all_params[param]])
        except:
            number[param] = 1
    num_combi = int(number.prod())
    print(num_combi, "iterações")
    
    vet_n_estimators = np.ones(num_combi,int)
    vet_best_acc = np.ones(num_combi)
    vet_mean_acc = np.ones(num_combi)
    
    combinacoes = possibilidades(number, num_var)
    combinacoes -= 1
    
    for comb in range(num_combi):
        posic = combinacoes.loc[comb]
        params = {}
        for param in range(num_var):
            try:
                params[all_params[param]] = multi_params[all_params[param]][posic[param]]
            except:
                params[all_params[param]] = multi_params[all_params[param]]
        gb = GradientBoostingClassifier(**params)
        bal_acc = cv_estimate(gb, params, X_train, y_train, kf)
        
        mean_acc = bal_acc.mean()
        best_acc = bal_acc.argmax()
        score_max = bal_acc[best_acc]
        vet_n_estimators[comb] = best_acc + 1
        vet_best_acc[comb] = score_max
        vet_mean_acc[comb] = mean_acc
        
    best = vet_mean_acc.argmax()
    
    posic = combinacoes.loc[best]
    params = {}
    for param in range(num_var):
        try:
            params[all_params[param]] = multi_params[all_params[param]][posic[param]]
        except:
            params[all_params[param]] = multi_params[all_params[param]]
    params['n_estimators'] = vet_n_estimators[best]
    return vet_n_estimators, vet_best_acc, vet_mean_acc,best, params