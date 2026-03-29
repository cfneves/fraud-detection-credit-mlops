"""
explainability/shap_report.py
------------------------------
Relatório de explicabilidade SHAP para modelos de crédito e fraude.

Gera:
- SHAP global: features mais importantes no modelo inteiro
- SHAP local: por que essa transação específica foi sinalizada como fraude?
- Relatório HTML por cliente/transação (para auditoria e conformidade regulatória)
"""

import os
import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


class SHAPReporter:
    """
    Gera análises SHAP globais e locais para modelos de fraude/crédito.

    Por que SHAP?
    - É o padrão da indústria financeira para explicabilidade
    - LGPD e regulações do Banco Central exigem explicação de decisões
    - Permite identificar quais features estão causando falsos positivos
    """

    def __init__(self, model, feature_names: List[str], output_dir: str = "reports"):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

    def fit_explainer(self, X_background: np.ndarray, sample_size: int = 500):
        """
        Inicializa o SHAP explainer com amostra de background.

        Para LightGBM/XGBoost: usa TreeExplainer (muito mais rápido).
        Para outros modelos: usa KernelExplainer (mais lento, universal).
        """
        print(f"Inicializando SHAP explainer...")

        try:
            # TreeExplainer é ordens de magnitude mais rápido para tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print("  Usando TreeExplainer (otimizado para gradient boosting)")
        except Exception:
            # Fallback para modelos não-tree
            background = shap.sample(X_background, sample_size)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
            print("  Usando KernelExplainer (universal)")

    def compute_shap_values(self, X: np.ndarray, max_samples: int = 1000) -> np.ndarray:
        """Calcula valores SHAP para um conjunto de amostras."""
        if self.explainer is None:
            raise RuntimeError("Execute fit_explainer() primeiro.")

        # Limitar amostras para eficiência
        n = min(len(X), max_samples)
        X_sample = X[:n]

        print(f"Calculando SHAP values para {n} amostras...")
        self.shap_values = self.explainer.shap_values(X_sample)

        # Para classificadores binários, pegar valores da classe positiva (fraude)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

        return self.shap_values

    def plot_global_importance(self, top_n: int = 20, save: bool = True):
        """
        Plot de importância global das features (SHAP mean |value|).

        Responde: "Quais features mais influenciam o modelo no geral?"
        """
        if self.shap_values is None:
            raise RuntimeError("Execute compute_shap_values() primeiro.")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False,
            plot_type="bar"
        )
        plt.title("Importância Global das Features (SHAP)", fontweight="bold")
        plt.tight_layout()

        if save:
            path = f"{self.output_dir}/figures/shap_global_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Salvo em: {path}")
        plt.close()

    def plot_beeswarm(self, X: np.ndarray, top_n: int = 20, save: bool = True):
        """
        Beeswarm plot: mostra direção e magnitude do impacto de cada feature.

        Responde: "Feature X alta aumenta ou diminui o risco de fraude?"
        """
        if self.shap_values is None:
            raise RuntimeError("Execute compute_shap_values() primeiro.")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False
        )
        plt.title("SHAP Beeswarm — Impacto e Direção por Feature", fontweight="bold")
        plt.tight_layout()

        if save:
            path = f"{self.output_dir}/figures/shap_beeswarm.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Salvo em: {path}")
        plt.close()

    def explain_single_transaction(
        self,
        transaction: np.ndarray,
        transaction_id: Optional[str] = None,
        y_true: Optional[int] = None,
        top_n: int = 10
    ) -> dict:
        """
        Explicação de uma transação individual.

        Responde: "Por que essa transação foi classificada como fraude?"

        Args:
            transaction: vetor de features da transação (1D)
            transaction_id: ID da transação (para relatório)
            y_true: label real (0=legítima, 1=fraude) se conhecido
        """
        if self.explainer is None:
            raise RuntimeError("Execute fit_explainer() primeiro.")

        transaction_2d = transaction.reshape(1, -1)
        shap_vals = self.explainer.shap_values(transaction_2d)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals_1d = shap_vals[0]
        proba = self._get_proba(transaction_2d)

        # Ordenar features por |SHAP value|
        indices = np.argsort(np.abs(shap_vals_1d))[::-1][:top_n]

        explanation = {
            "transaction_id": transaction_id or "N/A",
            "fraud_probability": float(proba),
            "y_true": y_true,
            "top_features": []
        }

        for idx in indices:
            explanation["top_features"].append({
                "feature": self.feature_names[idx],
                "value": float(transaction[idx]),
                "shap_value": float(shap_vals_1d[idx]),
                "direction": "aumenta risco" if shap_vals_1d[idx] > 0 else "reduz risco"
            })

        return explanation

    def generate_html_report(
        self,
        transaction: np.ndarray,
        transaction_id: str = "TX_001",
        y_true: Optional[int] = None
    ) -> str:
        """
        Gera relatório HTML por transação — para auditoria e conformidade.

        Em produção, esse relatório pode ser enviado para o time de risco
        ou usado como justificativa regulatória de bloqueio.
        """
        explanation = self.explain_single_transaction(
            transaction, transaction_id, y_true
        )

        proba = explanation["fraud_probability"]
        risk_level = (
            "ALTO" if proba >= 0.7 else
            "MÉDIO" if proba >= 0.4 else
            "BAIXO"
        )
        risk_color = "#e74c3c" if proba >= 0.7 else "#f39c12" if proba >= 0.4 else "#27ae60"

        rows = ""
        for feat in explanation["top_features"]:
            direction_color = "#e74c3c" if feat["shap_value"] > 0 else "#27ae60"
            rows += f"""
            <tr>
                <td>{feat['feature']}</td>
                <td>{feat['value']:.4f}</td>
                <td style="color:{direction_color}; font-weight:bold">
                    {feat['shap_value']:+.4f}
                </td>
                <td style="color:{direction_color}">{feat['direction']}</td>
            </tr>"""

        true_label_html = ""
        if y_true is not None:
            label_text = "FRAUDE ✗" if y_true == 1 else "LEGÍTIMA ✓"
            label_color = "#e74c3c" if y_true == 1 else "#27ae60"
            true_label_html = f"""
            <p><strong>Label Real:</strong>
               <span style="color:{label_color}">{label_text}</span>
            </p>"""

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório SHAP — {transaction_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .risk-box {{
            display: inline-block; padding: 10px 20px; border-radius: 5px;
            font-size: 1.4em; font-weight: bold; color: white;
            background-color: {risk_color}; margin: 15px 0;
        }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th {{ background-color: #2c3e50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .meta {{ color: #666; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>Relatório de Explicabilidade — Detecção de Fraude</h1>

    <p><strong>Transação:</strong> {transaction_id}</p>
    <p><strong>Probabilidade de Fraude:</strong> {proba:.1%}</p>
    <div class="risk-box">RISCO {risk_level}</div>
    {true_label_html}

    <h2>Principais Fatores de Decisão (SHAP)</h2>
    <p>Os valores SHAP indicam quanto cada feature contribuiu para aumentar (+)
       ou reduzir (−) a probabilidade de fraude nesta transação específica.</p>

    <table>
        <tr>
            <th>Feature</th>
            <th>Valor</th>
            <th>Contribuição SHAP</th>
            <th>Efeito</th>
        </tr>
        {rows}
    </table>

    <div class="meta">
        <p>Gerado automaticamente pelo pipeline de ML — crédito e fraude.</p>
        <p>Método: SHAP TreeExplainer | Modelo: Stacking Ensemble</p>
    </div>
</body>
</html>"""

        path = f"{self.output_dir}/{transaction_id}_shap_report.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Relatório HTML salvo em: {path}")
        return path

    def _get_proba(self, X: np.ndarray) -> float:
        """Obtém probabilidade da classe positiva."""
        try:
            proba = self.model.predict_proba(X)[0][1]
        except Exception:
            proba = float(self.model.predict(X)[0])
        return proba
