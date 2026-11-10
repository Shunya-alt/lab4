"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = np.array(data.train_features, dtype=object)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features, dtype=object)
    y_test = np.array(data.test_target)

    #предобработка: заполняем пропуски медианой и масштабируем признаки
    #делаем это через пайплайн, чтобы избежать утечки данных из теста
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), slice(0, X_train.shape[1]))
        ],
        remainder='passthrough'
    )
    
    #выбираем Random Forest 
    #ограничиваем глубину (max_depth=6), чтобы дерево не переобучилось, и ставим class_weight='balanced' 
    #для борьбы с дисбалансом классов (если одних больше, чем других)
    model = RandomForestClassifier(
        n_estimators=200,          
        max_depth=6,               
        min_samples_split=5,       
        class_weight='balanced',   
        random_state=data.random_seed,
        n_jobs=-1                  
    )
    
    #собираем пайплайн: данные сначала проходят через preprocessor, затем идут в модель
    pipeline = make_pipeline(preprocessor, model)
    
    #обучение на train-выборке (задание)
    pipeline.fit(X_train, y_train)
    
    #получаем вероятности положительного класса (нужны для ROC-AUC)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    #получаем бинарные предсказания (0 или 1)
    predictions = pipeline.predict(X_test)
    
    #считаем метрики
    roc_auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    return EnsembleResult(
        model_name="RandomForest (n_estimators=200, max_depth=6)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )
