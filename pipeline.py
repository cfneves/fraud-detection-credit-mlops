"""
pipeline.py
-----------
Pipeline principal: executa EDA, feature engineering, treino, avaliação e SHAP.

Uso:
    python pipeline.py --dataset fraud --quick
    python pipeline.py --dataset fraud --full
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Adicionar src/ ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.engineering import FraudFeatureEngineer, CreditScoringFeatureEngineer
from src.train import FraudModelTrainer
from src.metrics import full_evaluation_report
from src.shap_report import SHAPReporter


def load_fraud_dataset(data_dir: str = "data/raw/fraud") -> pd.DataFrame:
    """Carrega o dataset Credit Card Fraud Detection."""
    csv_path = f"{data_dir}/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset não encontrado em: {csv_path}")
        print("Execute: python src/data_download.py --dataset fraud")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Dataset carregado: {df.shape[0]:,} transações, {df.shape[1]} features")
    print(f"Fraudes: {df['Class'].sum():,} ({df['Class'].mean():.3%})")
    return df


def run_eda(df: pd.DataFrame, target_col: str = "Class"):
    """EDA básico — primeiros passos em qualquer projeto de fraude."""
    print("\n" + "="*50)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*50)

    # Desbalanceamento
    counts = df[target_col].value_counts()
    ratio = counts[0] / counts[1]
    print(f"\nDesbalanceamento:")
    print(f"  Classe 0 (legítima): {counts[0]:,}")
    print(f"  Classe 1 (fraude):   {counts[1]:,}")
    print(f"  Ratio:               {ratio:.0f}:1")

    # Estatísticas do valor da transação por classe
    if "Amount" in df.columns:
        print(f"\nValor médio da transação:")
        print(df.groupby(target_col)["Amount"].describe().round(2).to_string())

    # Valores ausentes
    missing = df.isnull().sum().sum()
    print(f"\nValores ausentes: {missing}")

    # Shape
    print(f"\nDimensões: {df.shape}")
    print(f"Memória: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


def run_pipeline(dataset: str = "fraud", quick_mode: bool = True):
    """Pipeline completo de treinamento e avaliação."""
    print("\n" + "="*60)
    print(f"  PIPELINE ML — CRÉDITO E FRAUDE")
    print(f"  Dataset: {dataset.upper()}")
    print("="*60)

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)

    # --- 1. Carregar dados ---
    if dataset == "fraud":
        df = load_fraud_dataset()
        target_col = "Class"
    else:
        print(f"Dataset '{dataset}' ainda não suportado neste pipeline.")
        print("Adicione o loader correspondente.")
        sys.exit(1)

    # --- 2. EDA ---
    run_eda(df, target_col)

    # --- 3. Feature Engineering ---
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)

    engineer = FraudFeatureEngineer()
    X, y = engineer.fit_transform(df, target_col=target_col, apply_scaling=True)

    print(f"\nFeatures geradas: {X.shape[1]}")
    print(f"Feature names: {engineer.feature_names_[:10]}...")

    # --- 4. Split treino/validação ---
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values,
        test_size=0.2,
        stratify=y.values,
        random_state=42
    )

    print(f"\nSplit:")
    print(f"  Treino:    {X_train.shape[0]:,} amostras")
    print(f"  Validação: {X_val.shape[0]:,} amostras")

    # --- 5. Treinar modelos ---
    print("\n" + "="*50)
    print("TREINAMENTO")
    print("="*50)

    # Quick mode: só LightGBM (mais rápido para teste)
    if quick_mode:
        from src.train import build_lightgbm, apply_resampling
        from sklearn.calibration import CalibratedClassifierCV

        X_res, y_res = apply_resampling(X_train, y_train, "smote")
        model = CalibratedClassifierCV(build_lightgbm(), cv=3, method="isotonic")
        model.fit(X_res, y_res)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_train_proba = model.predict_proba(X_train)[:, 1]

        results = full_evaluation_report(
            y_val, y_pred_proba,
            y_train_proba=y_train_proba,
            model_name="LightGBM (calibrado)",
            save_path="reports/figures/lightgbm_eval.png"
        )

    else:
        trainer = FraudModelTrainer()
        results = trainer.train_all(
            X_train, y_train, X_val, y_val,
            resample_strategy="smote",
            include_stacking=True
        )
        trainer.compare_results()
        model = trainer.models.get("Stacking") or trainer.models.get("LightGBM")
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    # --- 6. Explicabilidade SHAP ---
    print("\n" + "="*50)
    print("EXPLICABILIDADE SHAP")
    print("="*50)

    # Obter modelo base (sem calibrador) para SHAP TreeExplainer
    base_model = model
    if hasattr(model, "base_estimator"):
        base_model = model.base_estimator
    elif hasattr(model, "estimator"):
        base_model = model.estimator

    try:
        reporter = SHAPReporter(
            model=base_model,
            feature_names=engineer.feature_names_,
            output_dir="reports"
        )

        reporter.fit_explainer(X_train[:500])
        reporter.compute_shap_values(X_val[:500])
        reporter.plot_global_importance(top_n=15)
        reporter.plot_beeswarm(X_val[:500], top_n=15)

        # Gerar relatório HTML para a transação com maior probabilidade de fraude
        idx_max_fraud = np.argmax(y_pred_proba)
        reporter.generate_html_report(
            X_val[idx_max_fraud],
            transaction_id="TX_HIGH_RISK_001",
            y_true=int(y_val[idx_max_fraud])
        )

        print("\nExplicabilidade concluída.")

    except Exception as e:
        print(f"SHAP não disponível para este modelo: {e}")
        print("Instale: pip install shap")

    print("\n" + "="*60)
    print("  PIPELINE CONCLUÍDO")
    print(f"  Relatórios em: reports/")
    print(f"  Modelos em:    models/saved/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML — Crédito e Fraude")
    parser.add_argument("--dataset", default="fraud", choices=["fraud", "ieee", "credit_scoring"])
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Treina só LightGBM (rápido)")
    parser.add_argument("--full", action="store_true",
                        help="Treina todos os modelos + stacking")
    args = parser.parse_args()

    quick_mode = not args.full
    run_pipeline(dataset=args.dataset, quick_mode=quick_mode)


if __name__ == "__main__":
    main()
