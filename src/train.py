"""
models/train.py
---------------
Treinamento de modelos para crédito e detecção de fraude.

Cobre:
- Modelos individuais com otimização via Optuna
- Stacking ensemble (3 base learners + meta-learner)
- Calibração de probabilidade
- Persistência de modelos
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Funções de balanceamento
# ---------------------------------------------------------------------------

def apply_resampling(
    X: np.ndarray, y: np.ndarray, strategy: str = "smote", ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica técnica de rebalanceamento.

    Args:
        strategy: 'smote', 'smote_enn', 'adasyn', 'none'
        ratio: proporção desejada da classe minoritária
    """
    print(f"\nRebalanceamento: {strategy.upper()}")
    print(f"  Antes: {y.value_counts().to_dict() if hasattr(y, 'value_counts') else dict(zip(*np.unique(y, return_counts=True)))}")

    if strategy == "smote":
        sampler = SMOTE(sampling_strategy=ratio, random_state=42)
    elif strategy == "smote_enn":
        sampler = SMOTEENN(sampling_strategy=ratio, random_state=42)
    elif strategy == "adasyn":
        sampler = ADASYN(sampling_strategy=ratio, random_state=42)
    elif strategy == "none":
        return X, y
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    X_res, y_res = sampler.fit_resample(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"  Depois: {dict(zip(unique.tolist(), counts.tolist()))}")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Modelos individuais
# ---------------------------------------------------------------------------

def build_lightgbm(params: Optional[Dict] = None) -> LGBMClassifier:
    default = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default.update(params)
    return LGBMClassifier(**default)


def build_xgboost(params: Optional[Dict] = None) -> XGBClassifier:
    default = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 100,  # ajustar conforme ratio de fraude
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
    }
    if params:
        default.update(params)
    return XGBClassifier(**default)


def build_random_forest(params: Optional[Dict] = None) -> RandomForestClassifier:
    default = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default.update(params)
    return RandomForestClassifier(**default)


def build_catboost(params: Optional[Dict] = None) -> CatBoostClassifier:
    default = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",
        "random_state": 42,
        "verbose": 0,
    }
    if params:
        default.update(params)
    return CatBoostClassifier(**default)


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    use_calibration: bool = True,
    cv_folds: int = 5
) -> StackingClassifier:
    """
    Stacking ensemble com LightGBM, XGBoost e Random Forest como base learners
    e Logistic Regression como meta-learner.

    Por que stacking?
    - Cada modelo captura padrões diferentes
    - Meta-learner aprende a combinar as previsões de forma ótima
    - Mais robusto a overfitting do que um único modelo

    Por que LR como meta-learner?
    - Interpretável: os coeficientes mostram quanto cada modelo contribui
    - Não overfita nos dados de validação cruzada
    - Produz probabilidades bem calibradas
    """
    base_learners = [
        ("lgb", build_lightgbm()),
        ("xgb", build_xgboost()),
        ("rf", build_random_forest()),
    ]

    meta_learner = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,  # meta-learner recebe só previsões dos base learners
    )

    if use_calibration:
        return CalibratedClassifierCV(stacking, cv=3, method="isotonic")

    return stacking


# ---------------------------------------------------------------------------
# Otimização de hiperparâmetros com Optuna
# ---------------------------------------------------------------------------

def optimize_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    n_trials: int = 50, cv_folds: int = 5
) -> Dict:
    """
    Otimiza hiperparâmetros do LightGBM com Optuna.
    Métrica: PR-AUC (correta para dados desbalanceados).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Instale optuna: pip install optuna")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nMelhores parâmetros LightGBM:")
    print(f"  PR-AUC: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


# ---------------------------------------------------------------------------
# Pipeline completo de treinamento
# ---------------------------------------------------------------------------

class FraudModelTrainer:
    """Orquestra o treinamento, avaliação e persistência dos modelos."""

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models: Dict = {}
        self.results: Dict = {}

    def train_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        resample_strategy: str = "smote",
        include_stacking: bool = True
    ) -> Dict:
        """
        Treina todos os modelos e retorna resultados comparativos.
        """
        from src.metrics import full_evaluation_report

        # Rebalancear dados de treino
        X_res, y_res = apply_resampling(X_train, y_train, resample_strategy)

        # Modelos para treinar
        model_configs = {
            "LightGBM": build_lightgbm(),
            "XGBoost": build_xgboost(),
            "RandomForest": build_random_forest(),
        }

        if include_stacking:
            model_configs["Stacking"] = build_stacking_ensemble()

        for name, model in model_configs.items():
            print(f"\nTreinando: {name}...")
            model.fit(X_res, y_res)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_train_proba = model.predict_proba(X_train)[:, 1]

            results = full_evaluation_report(
                y_val, y_pred_proba,
                y_train_proba=y_train_proba,
                model_name=name,
                save_path=f"reports/figures/{name.lower()}_eval.png"
            )

            self.models[name] = model
            self.results[name] = results

            # Salvar modelo
            joblib.dump(model, f"{self.model_dir}/{name.lower()}.pkl")

        return self.results

    def compare_results(self) -> pd.DataFrame:
        """Tabela comparativa de todos os modelos treinados."""
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Modelo": name,
                "ROC-AUC": f"{res.get('roc_auc', 0):.4f}",
                "PR-AUC": f"{res.get('pr_auc', 0):.4f}",
                "KS": f"{res.get('ks_statistic', 0):.1f}",
                "Gini": f"{res.get('gini_coefficient', 0):.1f}",
                "Brier": f"{res.get('brier_score', 0):.4f}",
                "Threshold*": f"{res.get('optimal_threshold', 0.5):.4f}",
            })

        df = pd.DataFrame(rows)
        print("\nCOMPARAÇÃO DE MODELOS")
        print("=" * 70)
        print(df.to_string(index=False))
        print("* Threshold ótimo por custo de negócio")
        return df

    def load_model(self, model_name: str):
        """Carrega modelo salvo em disco."""
        path = f"{self.model_dir}/{model_name.lower()}.pkl"
        return joblib.load(path)
