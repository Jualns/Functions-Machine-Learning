import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

def train_grid(classifiers, params, X_train, y_train, X_test, y_test, skf, scoring, printar = True):
    
    #results = resultados numéricos das métricas para cada modelo, somente os melhores de cada classe
    results = {}
    #clfs = conjunto de parâmetros do melhor modelo por classe
    clfs = {}
    
    #pipeline é para rodar vários modelos com os mesmos dados e comparando os resultados baseado na métrica
    #gridsearch é para rodar um mesmo modelo com vários parâmetros diferentes e escolher o melhor deles baseado na métrica
    aux_stack = pd.DataFrame()
    aux_stack['y'] = y_test
    
    for clf in classifiers.keys():
        #if clf != 'svm':
        pipe = Pipeline([
            ('clf', classifiers[clf]())
        ])
        
        gs = GridSearchCV(pipe, params[clf], cv = skf, n_jobs = -1, scoring = scoring) #aplicamos o gridsearch dentro do pipeline 
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        
        results[clf] = (accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred))
        clfs[clf] = gs
        
        aux_stack[clf] = pd.DataFrame(gs.predict_proba(X_test))[1]
        if printar:
            print("----------------")
            print("Modelo ",clf)
            print(results[clf])
            print(classification_report(y_test, y_pred))
            print("----------------")
    
    return clfs, aux_stack

def skf_predict(skf, X, y, clfs, best_params, metric = balanced_accuracy_score):
    
    DataSet = pd.DataFrame(columns = clfs.keys())
    DataSet['y'] = y.copy()
    
    results = pd.DataFrame(columns = clfs.keys())
    results['fold'] = np.arange(skf.n_splits)
    k = 0 #fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        for model in clfs.keys():
            cl = clfs[model]()
            cl.set_params(**best_params[model])
            cl.fit(X_train, y_train)
            y_pred = cl.predict_proba(X_test)[:,1]
            DataSet.loc[test_index, model] = y_pred
            results.loc[k, model] = metric(y_test, cl.predict(X_test))
        
        k += 1
    return DataSet, results

def bests_params(clfs):
    melhores_params = {}
    for mod in clfs.keys():
        aux = {}
        for var in clfs[mod].best_params_.keys():
            i = var.split('__')[-1]
            aux[i] = clfs[mod].best_params_[var]
        melhores_params[mod] = aux
    return melhores_params
