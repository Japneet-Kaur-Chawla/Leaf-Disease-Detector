# mlmodel_training.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, StratifiedKFold, GridSearchCV)
from sklearn.metrics import balanced_accuracy_score

def train_rf_balanced(train_file, mask=None):
    arr = np.load(train_file)
    X = arr['X']
    y = arr['y']
    if mask is not None:
        X = X[:, mask.astype(bool)]
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='balanced_accuracy',n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    scores = cross_val_score(best_model, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    return best_model, float(scores.mean())


def evaluate_rf_once(model, eval_file, mask=None):
    d = np.load(eval_file)
    X = d['X']
    y = d['y']
    if mask is not None:
        X = X[:, mask.astype(bool)]
    pred = model.predict(X)
    return float(balanced_accuracy_score(y, pred))

def train_svm(train_file, mask=None):
    arr = np.load(train_file)
    X = arr['X']
    y = arr['y']
    if mask is not None:
        X = X[:, mask.astype(bool)]
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.001]
    }
    svm = SVC(kernel='rbf')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return best_model, float(scores.mean())

def evaluate_svm_once(model, eval_file, mask=None):
    d = np.load(eval_file)
    X = d['X']
    y = d['y']
    if mask is not None:
        X = X[:, mask.astype(bool)]
    pred = model.predict(X)
    return float((pred == y).mean())