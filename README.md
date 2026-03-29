# 🛡️ Detecção de Fraude em Cartão de Crédito com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange)]()

> Pipeline de Machine Learning de nível produção para detecção de fraude em cartões de crédito.
> Projeto de portfólio focado no mercado financeiro brasileiro.

---

## Resultado do Modelo

| Métrica | Valor | Benchmark |
|---|---|---|
| **ROC-AUC** | 0.9488 | > 0.85 |
| **PR-AUC** | 0.8789 | > 0.80 |
| **KS Statistic** | 89.7 | > 60 (excelente) |
| **Gini** | 89.8 | > 50 (excelente) |
| **Brier Score** | 0.0004 | < 0.10 (bem calibrado) |
| **F1 (threshold ótimo)** | 0.7964 | — |

---

## O Problema

Detecção de fraude em cartão de crédito é um problema de classificação binária com **desbalanceamento extremo**: em 284.807 transações, apenas 492 são fraudes (0,17%). Os desafios centrais são:

- **Desbalanceamento severo**: modelos ingênuos acertam 99.83% sem detectar nenhuma fraude
- **Custo assimétrico**: não detectar uma fraude (FN) custa muito mais do que um falso alarme (FP)
- **Necessidade de explicabilidade**: regulação exige justificativa para bloqueios (LGPD, Banco Central)
- **Calibração**: probabilidades devem ser confiáveis para precificação de risco

---

## Arquitetura do Projeto

```
projeto_ml_cred_fraude/
├── app.py                  # Dashboard Streamlit (interface didática)
├── pipeline.py             # Pipeline principal de ML
├── data_download.py        # Download de datasets via Kaggle API
├── requirements.txt        # Dependências Python
│
├── src/                    # Módulos de código
│   ├── __init__.py
│   ├── engineering.py      # Feature engineering (velocidade, desvio, tempo, rede)
│   ├── train.py            # Treinamento: LightGBM, XGBoost, RF, Stacking
│   ├── metrics.py          # Métricas de negócio: KS, Gini, PSI, custo
│   └── shap_report.py      # Explicabilidade SHAP + relatórios HTML
│
├── config/
│   └── config.yaml         # Configurações centralizadas
│
├── data/
│   └── raw/fraud/          # Dataset (não versionado no git)
│
├── models/saved/           # Modelos serializados (.pkl)
└── reports/figures/        # Gráficos gerados automaticamente
```

---

## Pipeline de ML

```
Dados Brutos (CSV)
      │
      ▼
Feature Engineering ──► Velocidade (1h, 6h, 24h, 7d)
      │                  Desvio (z-score, razão vs. histórico)
      │                  Tempo (hora, dia, noturno, fim de semana)
      │                  Rede (frequência cartão, combinação cartão+endereço)
      ▼
Balanceamento SMOTE (sampling_ratio=0.1)
      │
      ▼
Treinamento ──► LightGBM  ┐
               XGBoost    ├── Stacking Ensemble
               RandomForest┘   (meta: LogisticRegression)
      │
      ▼
Calibração (CalibratedClassifierCV, isotonic regression)
      │
      ▼
Avaliação: ROC-AUC, PR-AUC, KS, Gini, PSI, Custo de Negócio
      │
      ▼
Explicabilidade SHAP (global + local + HTML por transação)
```

---

## Decisões Técnicas

### Por que RobustScaler e não StandardScaler?
Transações financeiras têm outliers extremos (R$1 vs R$50.000 na mesma coluna). O `RobustScaler` usa mediana e IQR, sendo insensível a esses extremos. O `StandardScaler` seria distorcido pelos outliers.

### Por que PR-AUC como métrica principal?
Com 0.17% de fraudes, o ROC-AUC pode ser enganoso — um modelo que erra todas as fraudes ainda teria AUC alto. O PR-AUC foca na qualidade das predições positivas (fraudes), que é o que importa.

### Por que calibrar o modelo?
`LightGBM` e ensembles tendem a produzir probabilidades extremas (muito próximas de 0 ou 1). Para decisões de risco (limite de crédito, precificação), a probabilidade precisa ser interpretável. `CalibratedClassifierCV` com isotonic regression corrige isso.

### Por que SMOTE com `sampling_ratio=0.1`?
Oversamplear até 50/50 introduz muito ruído sintético. A proporção 10% equilibra o aprendizado da classe minoritária sem degradar a qualidade das amostras sintéticas.

---

## Métricas de Negócio

| Métrica | Fórmula | Uso |
|---|---|---|
| **KS Statistic** | max(cum_bad - cum_good) × 100 | Separação entre bons e maus pagadores |
| **Gini** | 2 × AUC - 1 | Padrão do mercado financeiro/risco de crédito |
| **PSI** | Σ(actual - expected) × ln(actual/expected) | Detecção de data drift em produção |
| **Brier Score** | mean((p - y)²) | Qualidade da calibração de probabilidade |

---

## Pré-requisitos

```bash
# Python 3.10+
pip install -r requirements.txt

# Credenciais Kaggle
# 1. Acesse https://www.kaggle.com/settings → Create New Token
# 2. Salve o kaggle.json em ~/.kaggle/kaggle.json
```

---

## Como Executar

### 1. Baixar os dados
```bash
python data_download.py --dataset fraud
```

### 2. Rodar o pipeline de treinamento
```bash
# Rápido (só LightGBM)
python pipeline.py --dataset fraud --quick

# Completo (todos os modelos + stacking)
python pipeline.py --dataset fraud --full
```

### 3. Abrir o dashboard interativo
```bash
streamlit run app.py
```

---

## Dashboard Streamlit

O dashboard foi desenvolvido com foco em **usabilidade para usuários não técnicos**. Cada página explica os conceitos em linguagem simples:

| Página | Conteúdo |
|---|---|
| 🏠 Início | Visão geral do projeto e o problema de fraude |
| 📊 Os Dados | EDA: distribuição, desbalanceamento, padrões temporais |
| 🧠 Como a IA Aprende | Feature engineering, LightGBM, calibração |
| 📈 Resultados | Métricas explicadas em linguagem simples + gráficos interativos |
| 🔍 Explicabilidade | SHAP: por que o modelo tomou essa decisão |
| 🎮 Simulador | Teste uma transação customizada em tempo real |

---

## Datasets

| Dataset | Fonte | Uso |
|---|---|---|
| Credit Card Fraud Detection | [Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Pipeline principal |
| IEEE-CIS Fraud Detection | [Kaggle (competição)](https://www.kaggle.com/competitions/ieee-fraud-detection) | Feature engineering avançado |
| Give Me Some Credit | [Kaggle (brycecf)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) | Credit scoring |

---

## Roadmap

- [x] Pipeline de ML com LightGBM calibrado
- [x] Métricas de negócio (KS, Gini, PSI, Brier)
- [x] Dashboard Streamlit didático
- [x] Explicabilidade com SHAP
- [ ] Stacking ensemble completo (LGB + XGB + RF + meta LR)
- [ ] Otimização Optuna (50 trials, PR-AUC)
- [ ] Autoencoder para detecção de anomalias não-supervisionada
- [ ] Monitoramento de drift com Evidently
- [ ] Deploy no Streamlit Cloud

---

## Tecnologias

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-00A3E0?style=flat)
![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600?style=flat)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![SHAP](https://img.shields.io/badge/-SHAP-FF4B4B?style=flat)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white&style=flat)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)

---

## Autor

**Cláudio Neves**
Cientista de Dados | Especialização em Crédito & Fraude

---

*Projeto desenvolvido para portfólio profissional — mercado financeiro brasileiro.*
