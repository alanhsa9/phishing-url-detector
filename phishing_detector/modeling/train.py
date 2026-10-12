import os
import argparse
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, precision_score, recall_score, fbeta_score, 
    ConfusionMatrixDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import warnings
from mlflow import MlflowClient

def load_data():
    """Descarga y carga el dataset de phishing."""
    phiusiil_phishing_url_website = fetch_ucirepo(id=967)
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets
    
    df = pd.concat([X, y], axis=1)
    target_col = y.columns[0]
    return df, target_col

def preprocess_data(df, target_col):
    """Aplica la limpieza y eliminación de variables innecesarias."""
    df_model = df.drop(columns=['FILENAME'], errors='ignore').copy()
    domain_groups = df_model['Domain']
    
    columnas_excluir = ['URL', 'Domain', 'TLD', 'Title']
    df_clean = df_model.drop(columns=columnas_excluir).copy()
    df_clean = df_clean.drop_duplicates()
    
    domain_groups = domain_groups.loc[df_clean.index]
    y_clean = df_clean[target_col]
    X_clean = df_clean.drop(columns=[target_col])

    return X_clean, y_clean, domain_groups

def split_data(X_clean, y_clean, domain_groups, random_state=42):
    """Divide los datos en Train, Validation y Test respetando los grupos."""
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_dev_idx, test_idx = next(sgkf_test.split(X_clean, y_clean, groups=domain_groups))
    
    X_dev = X_clean.iloc[train_dev_idx].copy()
    y_dev = y_clean.iloc[train_dev_idx].copy()
    groups_dev = domain_groups.iloc[train_dev_idx].copy()
    
    X_test = X_clean.iloc[test_idx].copy()
    y_test = y_clean.iloc[test_idx].copy()
    
    # Sincronizamos la semilla de validación conservando el comportamiento por defecto (123) 
    # cuando la semilla de entrada sea la estándar (42).
    val_random_state = 123 if random_state == 42 else random_state + 81
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=val_random_state)
    train_idx, val_idx = next(sgkf_val.split(X_dev, y_dev, groups=groups_dev))
    
    X_train = X_dev.iloc[train_idx].copy()
    y_train = y_dev.iloc[train_idx].copy()
    X_val = X_dev.iloc[val_idx].copy()
    y_val = y_dev.iloc[val_idx].copy()
    
    groups_train = groups_dev.iloc[train_idx].copy()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, groups_train

def train_model(X_train, y_train, n_estimators=200, max_depth=8, min_samples_split=10, random_state=42):
    """Entrena el modelo Random Forest"""
    print("Entrenando el modelo")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, groups_train,
                   n_estimators=200, max_depth=8, min_samples_split=10, random_state=42):
    """Calcula las métricas de evaluación, reentrena un modelo final y genera el reporte."""
    print("Evaluando modelo")
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    
    final_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        min_samples_leaf=5, 
        max_features='sqrt', 
        random_state=random_state, 
        n_jobs=-1
    )
    final_model.fit(X_train_final, y_train_final)
    
    y_pred = final_model.predict(X_test)
    
    test_precision = precision_score(y_test, y_pred, pos_label=0)
    test_recall = recall_score(y_test, y_pred, pos_label=0)
    test_f2 = fbeta_score(y_test, y_pred, beta=2, pos_label=0)
    
    # Configuración de directorio temporal/dedicado para artefactos
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reports_dir = os.path.join(base_dir, 'reports', 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, "classification_report.txt")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    pr_path = os.path.join(reports_dir, "pr_curve.png")
    
    # Generar y guardar el reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legítima (1)'])
    print("\n[Reporte de Clasificación]")
    print(report)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # Generar y guardar Matriz de Confusión
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['Phishing', 'Legítima'], cmap=plt.cm.Blues
    )
    cm_disp.figure_.savefig(cm_path)
    plt.close(cm_disp.figure_)
    
    # Generar y guardar Curva Precision-Recall
    y_proba_0 = final_model.predict_proba(X_test)[:, 0]
    pr_disp = PrecisionRecallDisplay.from_predictions(
        y_test, y_proba_0, pos_label=0, name="RF (Phishing)"
    )
    pr_disp.figure_.savefig(pr_path)
    plt.close(pr_disp.figure_)
    
    importances = final_model.feature_importances_
    df_importances = pd.DataFrame({'Variable': X_train_final.columns, 'Importancia': importances})
    df_importances = df_importances.sort_values(by='Importancia', ascending=False)
    print('\n[Top 10 variables más importantes]')
    print(df_importances.head(10).to_string(index=False))
    
    metrics = {
        'test_f2': test_f2,
        'test_precision': test_precision,
        'test_recall': test_recall
    }
    
    artifacts = {
        'report': report_path,
        'cm': cm_path,
        'pr': pr_path
    }
    
    return final_model, metrics, artifacts

def log_to_mlflow(final_model, metrics, artifacts, X_test, run_params):
    """Registra los parámetros, métricas y el modelo en MLflow."""
    print("\nRegistrando experimento en MLflow")

    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
    
    mlflow_uri = "http://127.0.0.1:8080"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    
    experiment_name = "deteccion_url"
    experiment_description = (
        "This is the phishing URL detection project. "
        "This experiment contains the baseline models for detecting malicious URLs."
    )
    
    experiment_tags = {
        "project_name": "deteccion_url",
        "dataset": "PhiUSIIL",
        "task": "phishing_url_detection",
        "model_type": "classification",
        "mlflow.note.content": experiment_description,
    }
    # Excepcion por si el servidor HTTP no está corriendo, crear el repo localmente.
    try:
        url_experiment = client.get_experiment_by_name(experiment_name)
        if url_experiment is None:
            client.create_experiment(name=experiment_name, tags=experiment_tags)
    except Exception as e:
        print(f"No se pudo contactar el servidor MLflow en {mlflow_uri}. Usando directorio local 'mlruns/'.")
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    
    run_name = 'url_rf_baseline'
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(run_params)
        mlflow.log_metrics(metrics)
        
        # Registro de los artefactos usando las rutas absolutas generadas
        mlflow.log_artifact(artifacts['report'])
        mlflow.log_artifact(artifacts['cm'])
        mlflow.log_artifact(artifacts['pr'])
        
        mlflow.sklearn.log_model(
            sk_model=final_model,
            input_example=X_test.head(5),
            name='url_rf_baseline_model',
            skops_trusted_types=['sklearn.tree._tree.Tree']
        )
        
        print(f"Experimento guardado exitosamente.")
        print(f"Run ID: {run.info.run_id}")

def parse_args():
    """Configura e interpreta los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenamiento parametrizado del modelo RandomForest para detección de Phishing.")
    parser.add_argument('--n_estimators', type=int, default=200, 
                        help='Número de árboles en el bosque (default: 200).')
    parser.add_argument('--max_depth', type=int, default=8, 
                        help='Profundidad máxima de los árboles (default: 8).')
    parser.add_argument('--min_samples_split', type=int, default=10, 
                        help='Número mínimo de muestras necesarias para dividir un nodo (default: 10).')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Semilla aleatoria para garantizar la reproducibilidad (default: 42).')
    return parser.parse_args()

def main():
    """Orquesta la ejecución completa del pipeline."""
    args = parse_args()
    
    df, target_col = load_data()
    X_clean, y_clean, domain_groups = preprocess_data(df, target_col)
    
    X_train, y_train, X_val, y_val, X_test, y_test, groups_train = split_data(
        X_clean, y_clean, domain_groups, random_state=args.random_state
    )
    
    model = train_model(
        X_train, y_train, 
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        min_samples_split=args.min_samples_split, 
        random_state=args.random_state
    )
    
    final_model, metrics, artifacts = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test, groups_train,
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        min_samples_split=args.min_samples_split, 
        random_state=args.random_state
    )
    
    run_params = {
        'model_algorithm': 'RandomForestClassifier',
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': args.random_state,
        'evaluation_split': 'StratifiedGroupKFold by Domain'
    }
    
    log_to_mlflow(final_model, metrics, artifacts, X_test, run_params)
    print("\nPipeline ejecutado correctamente")

if __name__ == "__main__":
    main()