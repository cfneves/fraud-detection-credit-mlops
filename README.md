# 🛡️ Detecção de Fraude em Cartão de Crédito com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)](https://lightgbm.readthedocs.io)
[![Streamlit App](https://img.shields.io/badge/Demo%20Live-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fraud-detection-credit-mlops-cfn.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Produção-brightgreen)]()

> Pipeline de Machine Learning de nível produção para detecção de fraude em cartões de crédito.
> Projeto de portfólio focado no mercado financeiro brasileiro.
>
> 🚀 **[Acesse o Dashboard Interativo](https://fraud-detection-credit-mlops-cfn.streamlit.app)**

---

## Por que este projeto existe

Toda vez que você passa o cartão, um sistema decide em menos de 300ms se aquela compra é legítima ou fraude. Esse sistema precisa ser preciso o suficiente para não bloquear sua compra no supermercado, mas sensível o suficiente para pegar o fraudador que clonout seu cartão às 3h da manhã.

Este projeto constrói exatamente esse sistema — do zero, com dados reais, seguindo as práticas usadas por bancos e fintechs brasileiras.

O objetivo não é só ter um modelo com métricas altas. É mostrar todo o raciocínio por trás: por que essas métricas, por que esse threshold, como explicar a decisão ao cliente, o que fazer quando o modelo erra.

---

## Quem usa sistemas como este

| Setor | Aplicação |
|---|---|
| Bancos (Itaú, Bradesco, C6, Nubank) | Aprovação de transações em tempo real |
| Adquirentes (Cielo, Stone, Rede) | Antifraude no processamento de pagamentos |
| Bureaus de crédito (Serasa, Boa Vista) | Score de risco para concessão de crédito |
| Seguradoras | Detecção de fraude em sinistros |
| E-commerce (Mercado Livre, Americanas) | Validação de pedidos antes do envio |

---

## O que o modelo decide na prática

O modelo não diz "fraude" ou "não fraude". Ele diz uma **probabilidade** — e o banco define o que fazer com ela:

| Score do modelo | Ação típica |
|---|---|
| Abaixo de 10% | Autoriza automaticamente |
| 10% a 30% | Monitora, registra para análise posterior |
| 30% a 70% | Solicita autenticação adicional (SMS, biometria) |
| Acima de 70% | Bloqueia e notifica o titular imediatamente |

Esses limites são ajustáveis. Um banco com clientes mais conservadores pode baixar o threshold. Uma fintech digital pode aceitar mais risco para não frustrar o usuário.

---

## Situações reais que este modelo detecta

**Teste de cartão roubado**
O fraudador faz uma compra de R$1,99 num app de streaming para ver se o cartão ainda funciona. Valores muito baixos, fora do padrão histórico do cliente, em horário incomum — esse padrão tem nome no mercado: *card testing*.

**Compra após vazamento de dados**
Dados de cartão são vazados e vendidos em lotes. O fraudador compra e testa muitos cartões em sequência. O modelo detecta pela velocidade: muitas transações no mesmo comerciante, em intervalo de minutos.

**Clonagem física**
O cartão é clonado num posto de gasolina. Horas depois, aparecem compras em outro estado ou país. O modelo compara localização, horário e valor com o histórico do cliente.

**Fraude de identidade (account takeover)**
Alguém acessa a conta do cliente e muda o endereço de entrega antes de fazer compras grandes. O modelo detecta pelo comportamento anômalo: primeiro acesso de IP diferente, depois compra de alto valor, tudo numa janela de minutos.

**Fraude amiga (friendly fraud)**
O próprio titular faz uma compra legítima e depois alega fraude para estornar. Mais difícil de detectar — exige análise comportamental histórica, não só a transação isolada.

---

## Por que o desbalanceamento é o verdadeiro problema

Em 284.807 transações, apenas 492 são fraudes. Isso parece bom até você perceber o que significa para um modelo de ML:

Se o modelo aprender a dizer "tudo é legítimo" para toda transação, ele acerta **99,83% das vezes**. Mas detecta **zero fraudes**.

Esse é o problema central. Resolver isso exige:
- Métricas certas (PR-AUC, KS, Gini — não acurácia)
- Balanceamento na hora do treinamento (SMOTE)
- Threshold ajustado pelo custo do erro, não por F1 genérico

O custo de uma fraude não detectada (R$100 em média) é 20x maior que o custo de um falso alarme (R$5 de operacional + atrito com o cliente). O modelo foi calibrado com essa assimetria em mente.

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

| Dataset | Fonte | Linhas | Uso |
|---|---|---|---|
| Credit Card Fraud Detection | [Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | **284.807** | Pipeline principal |
| IEEE-CIS Fraud Detection | [Kaggle (competição)](https://www.kaggle.com/competitions/ieee-fraud-detection) | ~590.000 | Feature engineering avançado |
| Give Me Some Credit | [Kaggle (brycecf)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) | 150.000 | Credit scoring |

### Por que o repositório contém apenas 2.000 linhas?

O dataset original **Credit Card Fraud Detection** possui **284.807 transações** e pesa **66 MB** — acima do limite recomendado para arquivos no GitHub (50 MB) e incompatível com o Streamlit Cloud, que não permite armazenar dados brutos no ambiente de deploy.

Para que o dashboard funcione online sem exigir credenciais do Kaggle, o repositório inclui `data/sample/creditcard_sample.csv`: uma **amostra sintética de 2.000 linhas** gerada com as mesmas propriedades estatísticas do dataset original (distribuição de valores, proporção de fraudes, correlações entre variáveis V1–V28).

> **Os resultados do modelo apresentados no dashboard (ROC-AUC 0.9488, KS 89.7, Gini 89.8) foram obtidos treinando no dataset completo de 284.807 transações**, rodando localmente com `python pipeline.py --dataset fraud --quick`. A amostra sintética serve apenas para visualização interativa no Streamlit Cloud.

Para reproduzir os resultados completos localmente:

```bash
# 1. Configure as credenciais do Kaggle (~/.kaggle/kaggle.json)
python data_download.py --dataset fraud   # baixa as 284.807 transações reais
python pipeline.py --dataset fraud --quick
```

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
- [x] Deploy no Streamlit Cloud — [fraud-detection-credit-mlops-cfn.streamlit.app](https://fraud-detection-credit-mlops-cfn.streamlit.app)

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

## Demo ao Vivo

[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)](https://fraud-detection-credit-mlops-cfn.streamlit.app)

Acesse o dashboard interativo e explore os resultados do modelo em tempo real:
**https://fraud-detection-credit-mlops-cfn.streamlit.app**

---

## Relevância para o Mercado Financeiro

Este projeto cobre os pontos que times de dados de bancos e fintechs brasileiras cobram em entrevistas:

**Métricas de risco de crédito** — KS Statistic e Gini são os indicadores padrão do mercado. ROC-AUC é o que a academia usa; KS é o que o gerente de risco entende. Saber a diferença importa.

**Calibração de probabilidade** — Bancos não querem apenas "fraude ou não". Querem saber *quanto* de risco. Uma probabilidade bem calibrada permite precificação, definição de limite e segmentação de clientes por faixa de risco.

**Explicabilidade (LGPD e BACEN)** — A Resolução 4.557 do Banco Central e a LGPD exigem que decisões automatizadas possam ser explicadas. SHAP resolve isso: o sistema consegue dizer ao cliente por que a transação foi bloqueada.

**Custo assimétrico de erro** — Em problemas reais, os dois tipos de erro não custam igual. FN (fraude não detectada) e FP (falso alarme) têm custos diferentes. O threshold do modelo foi otimizado com essa lógica, não pelo F1 genérico.

**Drift e monitoramento (PSI)** — Modelos degradam. O PSI (Population Stability Index) detecta quando a distribuição dos dados mudou o suficiente para exigir retreinamento. PSI = 0.0000 indica estabilidade total no dataset avaliado.

---

## Autor

**Cláudio Ferreira Neves**

Especialista em Ciência de Dados e Inteligência Artificial

Especialista em Business Intelligence, Big Data e Analytics

---

*Projeto desenvolvido para portfólio profissional — mercado financeiro brasileiro.*

---

📄 **[Casos de Uso detalhados → CASOS_DE_USO.md](CASOS_DE_USO.md)**
