import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def cargar_datos(filepath: str) -> pd.DataFrame:
    """Carga los datos iniciales desde la ruta especificada."""
    return pd.read_csv(filepath)

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas de negocio y elimina duplicados o inconsistencias.
    """
    df_clean = df.copy()
    
    # 1. Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # 2. Manejo de outliers / Reglas de negocio (ejemplo en datos médicos o numéricos)
    # Ejemplo: Filtrar edades no válidas si existieran
    if 'age' in df_clean.columns:
        df_clean = df_clean[df_clean['age'] > 0]
        
    return df_clean

def obtener_preprocesador(numeric_features: list, categorical_features: list):
    """
    Crea un Pipeline de scikit-learn para transformar columnas numéricas y categóricas.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

if __name__ == "__main__":
    # Prueba del script de preparación
    print("Ejecutando pipeline de limpieza de datos...")
    df_raw = cargar_datos("../datos/datos_ini/dataset.csv")
    df_processed = limpiar_datos(df_raw)
    df_processed.to_csv("../datos/datos_limp/dataset_clean.csv", index=False)
    print("Limpieza completada y datos guardados en datos/datos_limp/")
