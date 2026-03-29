"""
features/engineering.py
-----------------------
Feature engineering específico para crédito e detecção de fraude.

Cobre:
- Agregações temporais (janelas de 1h, 6h, 24h, 7d)
- Features comportamentais
- Detecção de velocidade de transações
- Encoding e scaling
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


class FraudFeatureEngineer:
    """
    Feature engineering para datasets de detecção de fraude.

    Focado no dataset IEEE-CIS e Credit Card Fraud (Kaggle).
    """

    def __init__(self, time_windows: List[int] = [1, 6, 24, 168]):
        """
        Args:
            time_windows: Janelas temporais em horas para agregações.
                         Default: [1h, 6h, 24h, 7 dias]
        """
        self.time_windows = time_windows
        self.scaler = RobustScaler()  # Robusto a outliers — importante em fraude
        self.feature_names_: List[str] = []

    def create_velocity_features(
        self, df: pd.DataFrame, amount_col: str = "TransactionAmt",
        time_col: str = "TransactionDT", card_col: str = "card1"
    ) -> pd.DataFrame:
        """
        Cria features de velocidade: número e valor de transações
        por janela temporal por cartão.

        Velocidade anômala é um dos sinais mais fortes de fraude.
        """
        df = df.copy()
        df_sorted = df.sort_values(time_col)

        for window in self.time_windows:
            window_sec = window * 3600

            # Número de transações na janela
            col_count = f"tx_count_{window}h"
            df_sorted[col_count] = (
                df_sorted.groupby(card_col)[time_col]
                .transform(lambda x: x.expanding().count())
            )

            # Valor acumulado na janela
            col_sum = f"tx_amount_sum_{window}h"
            df_sorted[col_sum] = (
                df_sorted.groupby(card_col)[amount_col]
                .transform(lambda x: x.expanding().sum())
            )

            # Valor médio na janela
            col_mean = f"tx_amount_mean_{window}h"
            df_sorted[col_mean] = (
                df_sorted.groupby(card_col)[amount_col]
                .transform(lambda x: x.expanding().mean())
            )

        return df_sorted

    def create_deviation_features(
        self, df: pd.DataFrame, amount_col: str = "TransactionAmt",
        card_col: str = "card1"
    ) -> pd.DataFrame:
        """
        Cria features de desvio em relação ao comportamento histórico.

        Um valor muito acima da média histórica do cartão é sinal de alerta.
        """
        df = df.copy()

        # Média e desvio padrão histórico por cartão
        card_stats = df.groupby(card_col)[amount_col].agg(["mean", "std"]).reset_index()
        card_stats.columns = [card_col, "card_mean_amount", "card_std_amount"]

        df = df.merge(card_stats, on=card_col, how="left")

        # Z-score do valor da transação em relação ao histórico do cartão
        df["amount_zscore"] = (
            (df[amount_col] - df["card_mean_amount"]) /
            (df["card_std_amount"].replace(0, 1))
        )

        # Razão entre valor atual e média histórica
        df["amount_ratio"] = df[amount_col] / df["card_mean_amount"].replace(0, 1)

        return df

    def create_time_features(
        self, df: pd.DataFrame, time_col: str = "TransactionDT"
    ) -> pd.DataFrame:
        """
        Extrai features temporais: hora, dia da semana, fim de semana.

        Fraudes têm padrões temporais distintos (ex: madrugada, feriados).
        """
        df = df.copy()

        # Assumindo que TransactionDT é segundos desde uma época
        # Converter para hora do dia e dia da semana
        seconds_per_day = 86400
        seconds_per_hour = 3600

        df["hour_of_day"] = (df[time_col] // seconds_per_hour) % 24
        df["day_of_week"] = (df[time_col] // seconds_per_day) % 7
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)

        return df

    def create_network_features(
        self, df: pd.DataFrame,
        card_cols: List[str] = ["card1", "card2", "card3", "card4", "card5", "card6"],
        address_cols: List[str] = ["addr1", "addr2"]
    ) -> pd.DataFrame:
        """
        Features de rede: quantas contas/endereços compartilham o mesmo dispositivo.

        Fraudes organizadas criam redes de contas — GNN detecta isso melhor,
        mas features simples já capturam parte do sinal.
        """
        df = df.copy()

        # Frequência de cada cartão (cartões com muitas transações são suspeitos)
        for col in card_cols:
            if col in df.columns:
                freq = df[col].map(df[col].value_counts())
                df[f"{col}_freq"] = freq

        # Combinação de cartão + endereço como chave de identidade
        if "card1" in df.columns and "addr1" in df.columns:
            df["card_addr_combo"] = (
                df["card1"].astype(str) + "_" + df["addr1"].astype(str)
            )
            df["card_addr_freq"] = df["card_addr_combo"].map(
                df["card_addr_combo"].value_counts()
            )
            df.drop(columns=["card_addr_combo"], inplace=True)

        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "isFraud",
        apply_scaling: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Aplica todo o pipeline de feature engineering.

        Args:
            df: DataFrame com dados brutos
            target_col: Nome da coluna alvo
            apply_scaling: Se True, aplica RobustScaler nas features numéricas

        Returns:
            X: Features processadas
            y: Target (ou None se não existir)
        """
        y = None
        if target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])

        # Aplicar transformações disponíveis
        if "TransactionDT" in df.columns:
            df = self.create_time_features(df)

        if "TransactionAmt" in df.columns and "card1" in df.columns:
            df = self.create_deviation_features(df)

        if "card1" in df.columns:
            df = self.create_network_features(df)

        # Remover colunas não numéricas para o modelo
        # (encoding de categorias — simplificado aqui com label encoding)
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes

        # Preencher NaN com mediana
        df = df.fillna(df.median(numeric_only=True))

        if apply_scaling:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        self.feature_names_ = df.columns.tolist()
        return df, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform sem fit — para dados de teste."""
        target_cols = ["isFraud", "Class"]
        for col in target_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes

        df = df.fillna(df.median(numeric_only=True))
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols] = self.scaler.transform(df[num_cols])

        return df


class CreditScoringFeatureEngineer:
    """
    Feature engineering para datasets de credit scoring.

    Focado no Give Me Some Credit (Kaggle) e German Credit (UCI).
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def create_debt_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features derivadas de razões de endividamento."""
        df = df.copy()

        # Razão dívida / renda mensal
        if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
            df["monthly_debt"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(
                df["MonthlyIncome"].median()
            )

        # Crédito rotativo usado em excesso
        if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            df["high_revolving_use"] = (
                df["RevolvingUtilizationOfUnsecuredLines"] > 0.7
            ).astype(int)

        return df

    def create_delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features consolidadas de inadimplência."""
        df = df.copy()

        delay_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]

        existing = [c for c in delay_cols if c in df.columns]
        if existing:
            df["total_past_due"] = df[existing].sum(axis=1)
            df["any_serious_delinquency"] = (df[existing[-1:]] > 0).astype(int)

        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Pipeline completo para credit scoring."""
        y = None
        if target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])

        # Remover index se existir
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df = self.create_debt_ratio_features(df)
        df = self.create_delinquency_features(df)

        # Tratar outliers extremos (winsorize nos percentis 1-99)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

        df = df.fillna(df.median(numeric_only=True))

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        return df, y
