import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Cargar datos procesados
ruta_datos = "../datos/datos_limp/dataset_clean.csv"
if not os.path.exists(ruta_datos):
    raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

df = pd.read_csv(ruta_datos)

# Separar variables predictoras y variable objetivo
X = df.drop(columns=["target"])  # Cambia 'target' si tu columna tiene otro nombre
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 2. Configurar MLflow
mlflow.set_experiment("Comparacion_Modelos_MLOps")

def evaluar_y_registrar(nombre_modelo, pipeline_base, param_grid):
    print(f"\n--- Entrenando {nombre_modelo} ---")
    
    grid = GridSearchCV(pipeline_base, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    preds_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    # Métricas
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, preds_proba) if preds_proba is not None else 0.0

    with mlflow.start_run(run_name=nombre_modelo):
        # Registrar hiperparámetros óptimos
        mlflow.log_params(grid.best_params_)
        
        # Registrar métricas
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        })
        
        # Registrar modelo evitando bloqueos de skops
        mlflow.sklearn.log_model(
            best_model, 
            artifact_path="model", 
            serialization_format="cloudpickle"
        )
        
    print(f"Éxito: {nombre_modelo} registrado. Accuracy: {acc:.4f} | F1: {f1:.4f}")

# 3. Modelos a comparar
# Modelo 1: Regresión Logística
pipe_lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression())])
grid_lr = {"clf__C": [0.1, 1.0, 10.0]}
evaluar_y_registrar("Logistic_Regression", pipe_lr, grid_lr)

# Modelo 2: Random Forest
pipe_rf = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
grid_rf = {"clf__n_estimators": [50, 100], "clf__max_depth": [3, 5, None]}
evaluar_y_registrar("Random_Forest", pipe_rf, grid_rf)
