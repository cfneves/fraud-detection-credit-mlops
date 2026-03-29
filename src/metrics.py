"""
evaluation/metrics.py
---------------------
Métricas de negócio para crédito e detecção de fraude.

Vai além do F1/AUC padrão — cobre o vocabulário que times de risco usam:
KS Statistic, Gini Coefficient, PSI, calibração, custo de negócio.
"""

import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")


def ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
    """
    KS Statistic (Kolmogorov-Smirnov).

    Mede a separação máxima entre as distribuições acumuladas
    de bons e maus pagadores.

    KS > 40: bom
    KS > 50: muito bom
    KS > 60: excelente (raro em dados reais)

    Returns:
        ks: valor do KS (0-100)
        threshold: ponto de corte onde o KS é máximo
    """
    df = pd.DataFrame({"score": y_pred_proba, "target": y_true})
    df = df.sort_values("score", ascending=False)

    n_bad = df["target"].sum()
    n_good = len(df) - n_bad

    df["cum_bad"] = df["target"].cumsum() / n_bad
    df["cum_good"] = (1 - df["target"]).cumsum() / n_good
    df["ks"] = abs(df["cum_bad"] - df["cum_good"])

    ks_value = df["ks"].max() * 100
    threshold = df.loc[df["ks"].idxmax(), "score"]

    return ks_value, threshold


def gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Gini Coefficient.

    Relação direta com AUC: Gini = 2 * AUC - 1
    Varia de -1 a 1 (quanto maior, mais discriminante).

    Gini > 0.3: aceitável
    Gini > 0.4: bom
    Gini > 0.5: muito bom
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return (2 * auc - 1) * 100


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """
    PSI (Population Stability Index).

    Mede se o perfil de entrada mudou entre treino e produção.
    Usado para detectar data drift antes de monitorar performance.

    PSI < 0.1:   estável — sem ação necessária
    PSI 0.1-0.2: monitorar com atenção
    PSI > 0.2:   instável — investigar e considerar retreino

    Args:
        expected: distribuição de referência (train)
        actual: distribuição atual (prod / validation)
        bins: número de faixas para discretização
    """
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Evitar divisão por zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )

    return psi


def business_cost_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray,
    fn_cost: float = 100.0, fp_cost: float = 5.0
) -> Tuple[float, float]:
    """
    Encontra o threshold ótimo minimizando custo de negócio.

    Em fraude, um falso negativo (fraude não detectada) custa muito mais
    do que um falso positivo (bloquear transação legítima).

    Args:
        fn_cost: custo de não detectar uma fraude (R$)
        fp_cost: custo de bloquear transação legítima (R$)

    Returns:
        optimal_threshold: ponto de corte ótimo
        min_cost: custo total mínimo (normalizado)
    """
    thresholds = np.linspace(0, 1, 200)
    costs = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        total_cost = (fn * fn_cost + fp * fp_cost) / len(y_true)
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]


def calibration_analysis(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> Dict:
    """
    Análise de calibração: o modelo diz 70% de probabilidade — é realmente 70%?

    Um modelo bem calibrado é essencial em crédito porque a probabilidade
    é usada para decisões de precificação e limite de crédito.

    Returns:
        dict com Brier Score e dados para o calibration plot
    """
    from sklearn.metrics import brier_score_loss

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )

    brier = brier_score_loss(y_true, y_pred_proba)

    return {
        "brier_score": brier,
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "calibration_quality": (
            "Bem calibrado" if brier < 0.1
            else "Calibração moderada" if brier < 0.2
            else "Mal calibrado — considere Platt Scaling"
        )
    }


def full_evaluation_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_train_proba: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
    fn_cost: float = 100.0,
    fp_cost: float = 5.0,
    save_path: Optional[str] = None
) -> Dict:
    """
    Relatório completo de avaliação para modelos de crédito/fraude.

    Combina métricas de ML com métricas de negócio.
    """
    print(f"\n{'='*60}")
    print(f"  RELATÓRIO DE AVALIAÇÃO: {model_name}")
    print(f"{'='*60}\n")

    results = {}

    # --- Métricas ML padrão ---
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    results["roc_auc"] = roc_auc
    results["pr_auc"] = pr_auc

    print(f"{'MÉTRICAS ML':}")
    print(f"  ROC-AUC:      {roc_auc:.4f}")
    print(f"  PR-AUC:       {pr_auc:.4f}  (métrica primária para dados desbalanceados)")

    # --- Métricas de negócio (vocabulário de risco) ---
    ks, ks_threshold = ks_statistic(y_true, y_pred_proba)
    gini = gini_coefficient(y_true, y_pred_proba)
    results["ks_statistic"] = ks
    results["gini_coefficient"] = gini

    print(f"\n{'MÉTRICAS DE RISCO':}")
    print(f"  KS Statistic: {ks:.2f}  {'[OK] Bom' if ks >= 40 else '[!] Baixo'}")
    print(f"  Gini:         {gini:.2f}  {'[OK] Bom' if gini >= 40 else '[!] Baixo'}")
    print(f"  KS Threshold: {ks_threshold:.4f}")

    # --- PSI (drift) ---
    if y_train_proba is not None:
        psi = population_stability_index(y_train_proba, y_pred_proba)
        results["psi"] = psi
        status = "[OK] Estavel" if psi < 0.1 else "[!] Monitorar" if psi < 0.2 else "[X] Instavel"
        print(f"\n{'ESTABILIDADE POPULACIONAL (PSI)':}")
        print(f"  PSI:          {psi:.4f}  {status}")

    # --- Threshold ótimo por custo de negócio ---
    opt_threshold, min_cost = business_cost_threshold(
        y_true, y_pred_proba, fn_cost, fp_cost
    )
    y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)
    f1_opt = f1_score(y_true, y_pred_opt)
    results["optimal_threshold"] = opt_threshold
    results["f1_at_optimal_threshold"] = f1_opt

    print(f"\n{'THRESHOLD ÓTIMO (custo de negócio)':}")
    print(f"  FN cost:      R$ {fn_cost:.0f} (fraude não detectada)")
    print(f"  FP cost:      R$ {fp_cost:.0f} (falso alarme)")
    print(f"  Threshold:    {opt_threshold:.4f}")
    print(f"  F1 no threshold: {f1_opt:.4f}")

    # --- Calibração ---
    calib = calibration_analysis(y_true, y_pred_proba)
    results["brier_score"] = calib["brier_score"]
    print(f"\n{'CALIBRAÇÃO':}")
    print(f"  Brier Score:  {calib['brier_score']:.4f}")
    print(f"  Status:       {calib['calibration_quality']}")

    print(f"\n{'='*60}\n")

    # --- Plot opcional ---
    if save_path:
        _plot_evaluation(y_true, y_pred_proba, results, calib, model_name, save_path)

    return results


def _plot_evaluation(y_true, y_pred_proba, results, calib, model_name, save_path):
    """Gera figura com 4 plots de avaliação."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Avaliação: {model_name}", fontsize=14, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, color="steelblue", lw=2,
                    label=f"ROC (AUC = {results['roc_auc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[0, 1].plot(recall, precision, color="darkorange", lw=2,
                    label=f"PR (AUC = {results['pr_auc']:.3f})")
    axes[0, 1].axhline(y=y_true.mean(), color="k", linestyle="--", lw=1,
                       label=f"Baseline ({y_true.mean():.3f})")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()

    # 3. Score Distribution
    axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6,
                    color="green", label="Legítima", density=True)
    axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6,
                    color="red", label="Fraude", density=True)
    axes[1, 0].axvline(x=results["optimal_threshold"], color="black",
                       linestyle="--", label=f"Threshold ({results['optimal_threshold']:.3f})")
    axes[1, 0].set_xlabel("Score de Probabilidade")
    axes[1, 0].set_ylabel("Densidade")
    axes[1, 0].set_title(f"Distribuição de Scores\n(KS={results['ks_statistic']:.1f}, Gini={results['gini_coefficient']:.1f})")
    axes[1, 0].legend()

    # 4. Calibration Plot
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="Calibração perfeita")
    axes[1, 1].plot(calib["mean_predicted_value"], calib["fraction_of_positives"],
                    "s-", color="steelblue", lw=2,
                    label=f"Modelo (Brier={results['brier_score']:.3f})")
    axes[1, 1].set_xlabel("Probabilidade prevista")
    axes[1, 1].set_ylabel("Frequência observada")
    axes[1, 1].set_title("Calibration Plot")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvo em: {save_path}")
