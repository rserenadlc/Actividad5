import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from datos_prep import cargar_datos, limpiar_datos, obtener_preprocesador

def evaluar_y_registrar(model_name, grid_search, X_test, y_test, experiment_name="Clasificacion_MLOps"):
    mlflow.set_experiment(experiment_name)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name=model_name):
        # Registrar Parámetros
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("algoritmo", model_name)
        
        # Registrar Métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Generar y guardar Matriz de Confusión como Artefacto
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        
        os.makedirs("temp_plots", exist_ok=True)
        plot_path = f"temp_plots/cm_{model_name}.png"
        plt.savefig(plot_path)
        plt.close()

        # Registrar Artefacto e Imagen en MLflow
        mlflow.log_artifact(plot_path, artifact_path="graficas")
        
        # Registrar Modelo
        mlflow.sklearn.log_model(best_model, artifact_path="modelo")

        print(f"\n--- {model_name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

def main():
    # 1. Cargar y Limpiar Datos
    ruta_datos = "../datos/datos_limp/dataset_clean.csv"
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}")
        return

    df = cargar_datos(ruta_datos)
    df = limpiar_datos(df)

    X = df.drop(columns=['target'])
    y = df['target']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = obtener_preprocesador(num_cols, cat_cols)

    # 2. Configurar Pipelines y Búsqueda de Hiperparámetros
    
    # Modelo 1: Regresión Logística
    pipe_lr = Pipeline(steps=[('prep', preprocessor), ('clf', LogisticRegression())])
    param_grid_lr = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs']
    }
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring='f1')
    grid_lr.fit(X_train, y_train)
    evaluar_y_registrar("Logistic_Regression", grid_lr, X_test, y_test)

    # Modelo 2: Random Forest
    pipe_rf = Pipeline(steps=[('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
    param_grid_rf = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='f1')
    grid_rf.fit(X_train, y_train)
    evaluar_y_registrar("Random_Forest", grid_rf, X_test, y_test)

if __name__ == "__main__":
    main()
