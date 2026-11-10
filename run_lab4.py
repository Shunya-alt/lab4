import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

def main():
    print("Загрузка данных german.csv...")
    #читаем данные 
    df = pd.read_csv("german.csv", sep=";")
    
    #отделяем целевую переменную от признаков
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    #приводим таргет к 0 и 1
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    elif y.min() == 1:
        y = y - 1
        
    #разбиваем на train/test 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    #создаем объект контракта
    data = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        random_seed=42
    )
    
    print("Обучение ансамбля...")
    result = run_ensemble_experiment(data)
    
    print("\nРЕЗУЛЬТАТЫ")
    print(f"Модель: {result.model_name}")
    print(f"ROC-AUC:  {result.roc_auc:.4f}  <-- проверь, чтобы было > 0.80!")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision:{result.precision:.4f}")
    print(f"Recall:   {result.recall:.4f}")
    print(f"F1-score: {result.f1:.4f}")

if __name__ == "__main__":
    main()