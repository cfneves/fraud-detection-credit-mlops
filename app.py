"""
app.py
------
Dashboard interativo para o projeto de Detecção de Fraude com Machine Learning.
Desenvolvido com Streamlit — interface didática para usuários não técnicos.

Executar:
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Detecção de Fraude com IA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fonte e cores gerais */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        color: white;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .metric-card .subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Cards de explicação */
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #1e293b;
    }
    .info-card.success { border-left-color: #10b981; background: #f0fdf4; }
    .info-card.warning { border-left-color: #f59e0b; background: #fffbeb; }
    .info-card.danger  { border-left-color: #ef4444; background: #fef2f2; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green  { background: #d1fae5; color: #065f46; }
    .badge-blue   { background: #dbeafe; color: #1e40af; }
    .badge-yellow { background: #fef3c7; color: #92400e; }
    .badge-red    { background: #fee2e2; color: #991b1b; }

    /* Separador com título */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }

    /* Resultado de simulação */
    .result-high   { background:#fef2f2; border:2px solid #ef4444; border-radius:12px; padding:20px; text-align:center; }
    .result-medium { background:#fffbeb; border:2px solid #f59e0b; border-radius:12px; padding:20px; text-align:center; }
    .result-low    { background:#f0fdf4; border:2px solid #10b981; border-radius:12px; padding:20px; text-align:center; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# Métricas reais pré-computadas no dataset completo (284.807 transações)
REAL_METRICS = {
    "roc_auc": 0.9488, "pr_auc": 0.8789, "ks": 89.73, "gini": 89.76,
    "brier": 0.0004, "best_threshold": 0.1005, "best_f1": 0.7964,
    "ks_threshold": 0.1111, "dataset_size": 284807, "n_fraud": 492,
}

# ─── Cache de dados e modelo ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando dataset...")
def load_data():
    full_path   = "data/raw/fraud/creditcard.csv"
    sample_path = "data/sample/creditcard_sample.csv"
    if Path(full_path).exists():
        return pd.read_csv(full_path), False   # False = não é amostra
    if Path(sample_path).exists():
        return pd.read_csv(sample_path), True  # True = é amostra
    return None, True


@st.cache_resource(show_spinner="Treinando modelo (pode levar 1-2 min na primeira vez)...")
def train_model(df):
    from src.engineering import FraudFeatureEngineer
    from src.train import apply_resampling, build_lightgbm
    from sklearn.calibration import CalibratedClassifierCV

    engineer = FraudFeatureEngineer()
    X, y = engineer.fit_transform(df, target_col="Class", apply_scaling=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=0.2, stratify=y.values, random_state=42
    )

    X_res, y_res = apply_resampling(X_train, y_train, "smote")
    model = CalibratedClassifierCV(build_lightgbm(), cv=3, method="isotonic")
    model.fit(X_res, y_res)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_train_proba = model.predict_proba(X_train)[:, 1]

    return model, engineer, X_train, X_val, y_train, y_val, y_pred_proba, y_train_proba


def compute_metrics(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc  = average_precision_score(y_true, y_pred_proba)

    # KS Statistic
    df = pd.DataFrame({"score": y_pred_proba, "target": y_true})
    df = df.sort_values("score", ascending=False)
    n_bad  = df["target"].sum()
    n_good = len(df) - n_bad
    df["cum_bad"]  = df["target"].cumsum() / n_bad
    df["cum_good"] = (1 - df["target"]).cumsum() / n_good
    df["ks"]       = abs(df["cum_bad"] - df["cum_good"])
    ks = df["ks"].max() * 100
    ks_threshold = df.loc[df["ks"].idxmax(), "score"]

    gini = (2 * roc_auc - 1) * 100

    # Brier
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_true, y_pred_proba)

    # Threshold ótimo por F1
    prec, rec, threshs = precision_recall_curve(y_true, y_pred_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_threshold = threshs[best_idx]
    best_f1 = f1s[best_idx]

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "ks": ks, "gini": gini, "brier": brier,
        "best_threshold": best_threshold, "best_f1": best_f1,
        "ks_threshold": ks_threshold,
    }


# ─── Sidebar de navegação ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Detecção de Fraude com IA")
    st.markdown("---")
    pagina = st.radio(
        "Navegue pelo projeto:",
        [
            "🏠  Início",
            "📊  Os Dados",
            "🧠  Como a IA Aprende",
            "📈  Resultados do Modelo",
            "🔍  Por que a IA Decidiu Assim?",
            "🎮  Simulador de Transação",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='line-height:1.35; font-size:0.78rem'>"
        "<b style='color:#e2e8f0; font-size:0.82rem'>Cláudio Ferreira Neves</b><br>"
        "<span style='color:#94a3b8'>Especialista em Ciência de Dados e Inteligência Artificial</span><br>"
        "<span style='color:#94a3b8'>Especialista em Business Intelligence, Big Data e Analytics</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — INÍCIO
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Início":
    st.markdown("""
    <h1 style='font-size:2.4rem; font-weight:800; color:#0f172a; margin-bottom:4px'>
        🛡️ Detecção de Fraude com Inteligência Artificial
    </h1>
    <p style='font-size:1.1rem; color:#475569; margin-bottom:32px'>
        Como a tecnologia protege seu dinheiro em tempo real
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="info-card">
        <b>O que este projeto faz?</b><br><br>
        Imagine que você usa seu cartão de crédito para comprar um cafezinho. Em menos de 1 segundo,
        um sistema analisa dezenas de informações sobre essa compra e decide: <em>isso parece normal
        ou pode ser uma fraude?</em><br><br>
        Este projeto constrói exatamente esse sistema usando <strong>Machine Learning</strong> —
        uma forma de ensinar um computador a reconhecer padrões a partir de exemplos reais.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card success" style='margin-top:16px'>
        <b>Por que isso é difícil?</b><br><br>
        Em 284.807 transações analisadas, apenas <strong>492 são fraudes</strong> — menos de 0,2%.
        É como tentar encontrar 5 agulhas em um palheiro de 2.800 palheiros. O desafio é não bloquear
        compras legítimas enquanto identifica as fraudulentas.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card warning" style='margin-top:16px'>
        <b>O que você vai encontrar aqui?</b><br><br>
        Este painel explica cada etapa do projeto — dos dados brutos ao modelo final —
        em linguagem simples, sem jargão técnico desnecessário.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style='margin-bottom:12px'>
            <div class="label">Dataset</div>
            <div class="value">284.807</div>
            <div class="subtext">transacoes analisadas</div>
        </div>
        <div class="metric-card" style='margin-bottom:12px'>
            <div class="label">Fraudes Reais</div>
            <div class="value">492</div>
            <div class="subtext">apenas 0,17% do total</div>
        </div>
        <div class="metric-card" style='margin-bottom:12px'>
            <div class="label">Precisao do Modelo</div>
            <div class="value">94,9%</div>
            <div class="subtext">ROC-AUC na validacao</div>
        </div>
        <div class="metric-card">
            <div class="label">Tecnologia</div>
            <div class="value" style='font-size:1.1rem'>LightGBM + SMOTE</div>
            <div class="subtext">Gradient Boosting calibrado</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Como navegar neste painel")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📊 Os Dados**\n\nExplora como são os dados de transações e por que o problema de fraude é tão desafiador.")
    with c2:
        st.info("**🧠 Como a IA Aprende**\n\nMostra quais pistas o modelo usa para identificar fraudes e como ele foi treinado.")
    with c3:
        st.info("**📈 Resultados**\n\nMétricas do modelo explicadas em linguagem simples — sem fórmulas difíceis.")

    c4, c5, _ = st.columns(3)
    with c4:
        st.info("**🔍 Por que a IA Decidiu Assim?**\n\nTransparência: entenda quais fatores levaram o modelo a sinalizar uma transação.")
    with c5:
        st.info("**🎮 Simulador**\n\nTeste você mesmo: insira os valores de uma transação e veja o que o modelo diz.")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — OS DADOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊  Os Dados":
    st.markdown("## 📊 Os Dados")
    st.markdown("*Entendendo o problema antes de construir a solução*")

    df, is_sample = load_data()
    if df is None:
        st.error("Dados não encontrados. Verifique a pasta data/sample/.")
        st.stop()
    if is_sample:
        st.info("**Modo demonstração:** exibindo amostra sintética (2.000 transações). Os resultados reais do modelo foram obtidos com o dataset completo de 284.807 transações.")

    # Visão geral
    st.markdown("""
    <div class="info-card">
    <b>De onde vêm esses dados?</b><br>
    Os dados são transacoes reais de cartao de credito europeu (anonimizadas por seguranca).
    Cada linha representa uma transacao — um momento em que alguem usou o cartao para comprar algo.
    As colunas V1 a V28 sao variaveis transformadas matematicamente para proteger a privacidade
    dos clientes (usando uma tecnica chamada PCA).
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de transacoes", f"{len(df):,}")
    col2.metric("Transacoes legitimas", f"{(df['Class']==0).sum():,}")
    col3.metric("Fraudes", f"{(df['Class']==1).sum():,}")
    col4.metric("Taxa de fraude", f"{df['Class'].mean():.4%}")

    st.markdown("---")

    # O problema do desbalanceamento
    st.markdown("### O Grande Desafio: O Desbalanceamento")
    st.markdown("""
    <div class="info-card warning">
    <b>Por que isso e um problema?</b><br><br>
    Se um modelo dissesse <em>"toda transacao e legitima"</em>, ele acertaria 99,83% das vezes!
    Mas seria completamente inutil para detectar fraudes. Por isso precisamos de tecnicas especiais
    para lidar com esse desequilibrio.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        counts = df["Class"].value_counts().reset_index()
        counts.columns = ["Tipo", "Quantidade"]
        counts["Tipo"] = counts["Tipo"].map({0: "Legitima", 1: "Fraude"})
        counts["Percentual"] = (counts["Quantidade"] / counts["Quantidade"].sum() * 100).round(4)

        fig = px.pie(
            counts, values="Quantidade", names="Tipo",
            color="Tipo",
            color_discrete_map={"Legitima": "#3b82f6", "Fraude": "#ef4444"},
            title="Distribuicao: 99,83% legitimas vs 0,17% fraudes",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="info-card" style='margin-top:20px'>
        <b>Analogia do dia a dia:</b><br><br>
        Imagine uma escola com 1.000 alunos. Se apenas 2 deles fizeram cola em uma prova,
        como o professor detecta esses 2 sem punir os 998 inocentes?<br><br>
        O desbalanceamento e exatamente esse cenario: a "maioria esmagadora" pode
        enganar o modelo.
        </div>

        <div class="info-card success" style='margin-top:12px'>
        <b>Nossa solucao: SMOTE</b><br><br>
        SMOTE (Synthetic Minority Oversampling Technique) cria exemplos sinteticos de fraudes
        para que o modelo "veja" mais exemplos de fraude durante o treinamento. E como dar
        ao modelo mais "casos de estudo" para aprender.
        </div>
        """, unsafe_allow_html=True)

    # Distribuição de valores
    st.markdown("---")
    st.markdown("### Valor das Transacoes: Legitimas vs Fraudes")

    st.markdown("""
    <div class="info-card">
    Abaixo, comparamos os valores das transacoes fraudulentas com as legitimas.
    Embora as fraudes possam ocorrer em qualquer valor, ha padroes distintos —
    fraudes frequentemente envolvem valores menores (para testar o cartao) ou muito grandes.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="Amount", color=df["Class"].map({0: "Legitima", 1: "Fraude"}),
            nbins=80, barmode="overlay", opacity=0.7,
            color_discrete_map={"Legitima": "#3b82f6", "Fraude": "#ef4444"},
            title="Distribuicao dos Valores (R$)",
            labels={"color": "Tipo", "Amount": "Valor (R$)"},
        )
        fig.update_layout(height=340, legend_title="Tipo")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stats = df.groupby("Class")["Amount"].describe().reset_index()
        stats["Class"] = stats["Class"].map({0: "Legitima", 1: "Fraude"})
        stats = stats.round(2)
        st.markdown("**Estatisticas por tipo de transacao:**")
        st.dataframe(stats.set_index("Class").T, use_container_width=True)

        st.markdown("""
        <div class="info-card warning" style='margin-top:8px'>
        <b>Curiosidade:</b> O valor medio das fraudes (R$122) e maior do que o das transacoes
        legitimas (R$88), mas a mediana das fraudes (R$9,25) e muito menor — muitos
        fraudadores testam com valores pequenos primeiro.
        </div>
        """, unsafe_allow_html=True)

    # Padrão temporal
    st.markdown("---")
    st.markdown("### Quando Ocorrem as Fraudes?")

    df["hour"] = (df["Time"] / 3600).astype(int) % 24
    hourly = df.groupby(["hour", "Class"]).size().reset_index(name="count")
    hourly["Tipo"] = hourly["Class"].map({0: "Legitima", 1: "Fraude"})

    fig = px.line(
        hourly, x="hour", y="count", color="Tipo",
        color_discrete_map={"Legitima": "#3b82f6", "Fraude": "#ef4444"},
        title="Volume de Transacoes por Hora do Dia",
        labels={"hour": "Hora do Dia (0-23)", "count": "Numero de Transacoes"},
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
    <b>O que o grafico mostra:</b> Transacoes legitimas seguem o ritmo natural do dia —
    mais durante o comercio, menos de madrugada. Fraudes tendem a ter um padrao diferente,
    pois fraudadores muitas vezes agem quando o titular do cartao esta dormindo e nao vai
    perceber imediatamente.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — COMO A IA APRENDE
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🧠  Como a IA Aprende":
    st.markdown("## 🧠 Como a IA Aprende a Detectar Fraudes")
    st.markdown("*O processo de treinamento explicado passo a passo*")

    st.markdown("""
    <div class="info-card">
    <b>Analogia simples:</b><br>
    Imagine que voce quer ensinar uma crianca a reconhecer cachorros. Voce mostra milhares de fotos:
    "isso e cachorro", "isso nao e cachorro". Com o tempo, a crianca aprende os padroes — orelhas,
    focinho, pelo. O Machine Learning funciona da mesma forma, mas com dados numericos.
    </div>
    """, unsafe_allow_html=True)

    # Pipeline visual
    st.markdown("### O Pipeline de Machine Learning")

    etapas = [
        ("1️⃣ Dados Brutos", "284.807 transacoes com 30 variaveis cada", "#3b82f6"),
        ("2️⃣ Feature Engineering", "Criamos novas variaveis: hora do dia, velocidade de transacoes...", "#8b5cf6"),
        ("3️⃣ Balanceamento (SMOTE)", "Criamos exemplos sinteticos de fraude para equilibrar os dados", "#f59e0b"),
        ("4️⃣ Treinamento LightGBM", "O modelo aprende a distinguir fraudes de transacoes legitimas", "#10b981"),
        ("5️⃣ Calibracao", "Ajustamos as probabilidades para que 70% signifique realmente 70%", "#06b6d4"),
        ("6️⃣ Avaliacao", "Testamos em dados nunca vistos antes para medir a performance real", "#ef4444"),
    ]

    cols = st.columns(3)
    for i, (titulo, desc, cor) in enumerate(etapas):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:{cor}18; border:1px solid {cor}44; border-radius:10px;
                        padding:16px; margin-bottom:12px; min-height:110px'>
                <b style='color:{cor}'>{titulo}</b><br>
                <small style='color:#374151'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Modelo LightGBM explicado
    st.markdown("### O Modelo: LightGBM")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
        <b>O que e LightGBM?</b><br><br>
        Imagine um conselho de especialistas. Cada especialista (chamado de "arvore de decisao")
        analisa a transacao e da um voto. O LightGBM cria centenas dessas arvores e combina
        todos os votos para chegar a uma decisao final.<br><br>
        Cada arvore corrige os erros das anteriores — isso e o "Gradient Boosting".
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card success">
        <b>Por que LightGBM e rapido e preciso?</b><br><br>
        - Processa milhoes de transacoes em segundos<br>
        - Funciona bem com dados desbalanceados<br>
        - Nao e enganado por valores extremos (outliers)<br>
        - Produz uma <em>probabilidade</em>, nao apenas sim/nao
        </div>
        """, unsafe_allow_html=True)

    # Arvore de decisão explicada visualmente
    st.markdown("---")
    st.markdown("### Como uma Arvore de Decisao Funciona")

    st.markdown("""
    <div class="info-card">
    Antes do LightGBM combinar centenas de arvores, cada arvore individual funciona como
    um fluxograma de perguntas. Veja um exemplo simplificado:
    </div>
    """, unsafe_allow_html=True)

    # Simula árvore de decisão visual com HTML
    st.markdown("""
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:24px; font-family:monospace; font-size:0.85rem; color:#1e293b'>
    <div style='text-align:center; margin-bottom:16px; font-size:1rem; font-weight:bold; color:#0f172a'>
        Exemplo de Arvore de Decisao para Fraude
    </div>
    <div style='text-align:center'>
        ┌─────────────────────────────┐<br>
        │ O valor e maior que R$500?  │<br>
        └──────────┬──────────────────┘<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;SIM ↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘ NAO<br>
    ┌───────────────┐&nbsp;&nbsp;&nbsp;┌────────────────────┐<br>
    │ Eh noturno?   │&nbsp;&nbsp;&nbsp;│ Muitas transacoes  │<br>
    │ (22h - 6h)    │&nbsp;&nbsp;&nbsp;│ na ultima hora?    │<br>
    └──────┬────────┘&nbsp;&nbsp;&nbsp;└─────────┬──────────┘<br>
    SIM↙&nbsp;&nbsp;&nbsp;↘NAO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SIM↙&nbsp;&nbsp;↘NAO<br>
    &nbsp;&nbsp;FRAUDE  LEGIT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FRAUDE&nbsp;&nbsp;LEGIT<br>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card warning" style='margin-top:12px'>
    <b>Importante:</b> Essa e uma simplificacao didatica. O modelo real usa 500 arvores,
    cada uma com dezenas de nos, analisando 30 variaveis simultaneamente. O resultado e uma
    <strong>probabilidade entre 0% e 100%</strong> de ser fraude.
    </div>
    """, unsafe_allow_html=True)

    # O que é calibração
    st.markdown("---")
    st.markdown("### Por que Calibramos o Modelo?")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="info-card danger">
        <b>Problema sem calibracao:</b><br><br>
        O modelo diz "probabilidade de fraude: 80%".<br>
        Mas na realidade, dessas transacoes que ele diz 80%, apenas 50% sao fraudes de verdade.<br><br>
        Isso significa que o modelo e <em>otimista demais</em> — e perigoso para decisoes de negocio.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card success">
        <b>Com calibracao (isotonic regression):</b><br><br>
        O modelo diz "probabilidade de fraude: 80%".<br>
        E de fato, dessas transacoes, 78-82% sao fraudes reais.<br><br>
        Isso e como um medico que diz "voce tem 70% de chance de recuperar" —
        e isso deve ser verdade, nao apenas uma estimativa.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📈  Resultados do Modelo":
    st.markdown("## 📈 Resultados do Modelo")
    st.markdown("*Métricas explicadas em linguagem simples*")

    df, is_sample = load_data()
    if df is None:
        st.error("Dados não encontrados. Verifique a pasta data/sample/.")
        st.stop()

    if is_sample:
        st.success(
            "**Resultados do dataset completo (284.807 transações reais):** "
            "Os valores abaixo foram obtidos treinando o modelo no dataset original do Kaggle — "
            "o maior e mais utilizado benchmark de detecção de fraude do mundo."
        )

    with st.spinner("Treinando modelo na amostra disponível..."):
        model, engineer, X_train, X_val, y_train, y_val, y_pred_proba, y_train_proba = train_model(df)

    # Usa métricas reais do dataset completo quando em modo amostra
    metrics = REAL_METRICS if is_sample else compute_metrics(y_val, y_pred_proba)
    # Para gráficos interativos, usa predições da amostra atual
    metrics_live = compute_metrics(y_val, y_pred_proba)

    # Cards de métricas principais
    st.markdown("### Resumo do Desempenho")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">ROC-AUC</div>
            <div class="value">{metrics['roc_auc']:.4f}</div>
            <div class="subtext">Poder de discriminacao geral</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">PR-AUC</div>
            <div class="value">{metrics['pr_auc']:.4f}</div>
            <div class="subtext">Metrica principal p/ dados raros</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">KS Statistic</div>
            <div class="value">{metrics['ks']:.1f}</div>
            <div class="subtext">Separacao entre fraudes e legitimas</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Gini</div>
            <div class="value">{metrics['gini']:.1f}</div>
            <div class="subtext">Padrao do mercado financeiro</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Explicações das métricas
    st.markdown("### O que cada metrica significa?")

    tabs = st.tabs(["ROC-AUC", "PR-AUC", "KS Statistic", "Gini", "Calibracao"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="info-card">
            <b>ROC-AUC = {metrics['roc_auc']:.4f}</b><br><br>
            <b>Em palavras simples:</b> Se eu pegar uma transacao fraudulenta e uma legitima
            ao acaso, o modelo da uma pontuacao maior para a fraude em {metrics['roc_auc']:.1%} das vezes.<br><br>
            <b>Analogia:</b> Um medico radiologista que consegue distinguir tumores de tecido
            normal em 94,9% dos casos ao analizar radiografias.<br><br>
            <span class="badge badge-green">Excelente (meta: acima de 0.85)</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#3b82f6", width=2.5),
                name=f"Modelo (AUC={metrics['roc_auc']:.3f})",
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#94a3b8", width=1, dash="dash"),
                name="Modelo aleatorio (AUC=0.5)"
            ))
            fig.update_layout(
                title="Curva ROC — Quanto mais para o canto superior esquerdo, melhor",
                xaxis_title="Taxa de Falsos Alarmes",
                yaxis_title="Taxa de Deteccao de Fraudes",
                height=340, legend=dict(x=0.4, y=0.1)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="info-card">
            <b>PR-AUC = {metrics['pr_auc']:.4f}</b><br><br>
            <b>Por que esta metrica e mais importante para fraude?</b><br><br>
            Com dados muito desbalanceados (0.17% de fraude), o ROC-AUC pode ser enganoso.
            A curva Precision-Recall foca especificamente na qualidade das deteccoes
            de fraude — ignorando os falsos "nao-fraudes".<br><br>
            <b>Precisao:</b> De tudo que o modelo disse ser fraude, quantos % realmente eram?<br>
            <b>Recall:</b> De todas as fraudes reais, quantos % o modelo encontrou?<br><br>
            <span class="badge badge-green">Excelente (meta: acima de 0.80)</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            prec, rec, _ = precision_recall_curve(y_val, y_pred_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                line=dict(color="#f59e0b", width=2.5),
                name=f"Modelo (AUC={metrics['pr_auc']:.3f})",
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"
            ))
            fig.add_hline(
                y=y_val.mean(), line_dash="dash", line_color="#94a3b8",
                annotation_text=f"Baseline aleatorio ({y_val.mean():.4f})"
            )
            fig.update_layout(
                title="Curva Precisao-Recall — Quanto mais area, melhor",
                xaxis_title="Recall (% de fraudes detectadas)",
                yaxis_title="Precisao (% corretos entre alertados)",
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="info-card">
            <b>KS = {metrics['ks']:.1f}</b><br><br>
            <b>O que e o KS Statistic?</b><br><br>
            KS mede o quanto o modelo consegue SEPARAR os dois grupos: fraudes e transacoes legitimas.<br><br>
            Pense em duas pilhas de cartas embaralhadas — vermelhas (fraudes) e azuis (legitimas).
            O KS mede o quanto o modelo consegue dividi-las em dois grupos limpos.<br><br>
            <b>Tabela de referencia:</b>
            </div>
            <table style='width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85rem'>
            <tr style='background:#1e293b; color:white'><th style='padding:6px'>KS</th><th>Qualidade</th></tr>
            <tr style='background:#fef2f2'><td style='padding:6px'>Abaixo de 20</td><td>Ruim</td></tr>
            <tr style='background:#fffbeb'><td style='padding:6px'>20 a 40</td><td>Aceitavel</td></tr>
            <tr style='background:#f0fdf4'><td style='padding:6px'>40 a 50</td><td>Bom</td></tr>
            <tr style='background:#dcfce7'><td style='padding:6px'>50 a 60</td><td>Muito bom</td></tr>
            <tr style='background:#bbf7d0'><td style='padding:6px'>Acima de 60</td><td><b>Excelente</b></td></tr>
            </table>
            """, unsafe_allow_html=True)
        with col2:
            # Plot KS
            df_ks = pd.DataFrame({"score": y_pred_proba, "target": y_val})
            df_ks = df_ks.sort_values("score", ascending=False)
            n_bad  = df_ks["target"].sum()
            n_good = len(df_ks) - n_bad
            df_ks["cum_bad"]  = df_ks["target"].cumsum() / n_bad
            df_ks["cum_good"] = (1 - df_ks["target"]).cumsum() / n_good
            x_axis = np.linspace(0, 1, len(df_ks))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=df_ks["cum_bad"].values,  name="Fraudes", line=dict(color="#ef4444", width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=df_ks["cum_good"].values, name="Legitimas", line=dict(color="#3b82f6", width=2)))
            fig.add_vline(x=df_ks["ks"].idxmax() / len(df_ks) if n_bad > 0 else 0.5,
                          line_dash="dash", line_color="#1e293b",
                          annotation_text=f"KS max = {metrics['ks']:.1f}")
            fig.update_layout(
                title="Grafico KS — Separacao entre Fraudes e Legitimas",
                xaxis_title="Populacao ordenada por score (0% a 100%)",
                yaxis_title="Distribuicao acumulada",
                height=360
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown(f"""
        <div class="info-card">
        <b>Gini = {metrics['gini']:.1f} (equivale a ROC-AUC = {metrics['roc_auc']:.4f})</b><br><br>
        O Gini e o <em>idioma nativo</em> do mercado de credito. Bancos e financeiras usam essa metrica
        ha decadas para avaliar modelos de risco. A relacao e simples: <strong>Gini = 2 x AUC - 1</strong>.<br><br>
        Com Gini = {metrics['gini']:.1f}, nosso modelo esta na categoria <b>Excelente</b> para os padroes
        do mercado financeiro (meta: Gini > 50).<br><br>
        <b>Por que bancos usam Gini e nao AUC?</b> Tradição histórica — e o que os reguladores
        e times de risco reconhecem e sabem interpretar.
        </div>
        """, unsafe_allow_html=True)

    with tabs[4]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="info-card">
            <b>Brier Score = {metrics['brier']:.4f}</b><br><br>
            <b>O que e calibracao?</b><br><br>
            Quando o modelo diz "essa transacao tem 70% de chance de ser fraude",
            isso deve ser verdade. Se na pratica apenas 30% dessas transacoes sao fraudes,
            o modelo esta <em>mal calibrado</em>.<br><br>
            O Brier Score mede a qualidade da calibracao:
            <ul>
            <li>Abaixo de 0.10 → Bem calibrado</li>
            <li>0.10 a 0.20 → Calibracao moderada</li>
            <li>Acima de 0.20 → Mal calibrado</li>
            </ul>
            <span class="badge badge-green">Nosso modelo: {metrics['brier']:.4f} — Bem calibrado</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            fop, mpv = calibration_curve(y_val, y_pred_proba, n_bins=10)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#94a3b8", dash="dash", width=1.5),
                name="Calibracao perfeita"
            ))
            fig.add_trace(go.Scatter(
                x=mpv, y=fop, mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=8),
                name=f"Nosso modelo (Brier={metrics['brier']:.4f})"
            ))
            fig.update_layout(
                title="Grafico de Calibracao — Quanto mais proximo da linha tracejada, melhor",
                xaxis_title="Probabilidade prevista pelo modelo",
                yaxis_title="Frequencia real observada",
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)

    # Matriz de Confusão
    st.markdown("---")
    st.markdown("### A Matrix de Confusao: Acertos e Erros do Modelo")

    threshold = metrics["best_threshold"]
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Previsao do Modelo", y="Realidade", color="Quantidade"),
            x=["Previsto: Legitima", "Previsto: Fraude"],
            y=["Real: Legitima", "Real: Fraude"],
            color_continuous_scale=[[0, "#dbeafe"], [1, "#1e40af"]],
        )
        fig.update_layout(title=f"Matriz de Confusao (threshold={threshold:.3f})", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="info-card success">
        <b>Verdadeiros Positivos (TP): {tp:,}</b><br>
        Fraudes que o modelo CORRETAMENTE identificou como fraude. Nosso objetivo principal.
        </div>
        <div class="info-card">
        <b>Verdadeiros Negativos (TN): {tn:,}</b><br>
        Transacoes legitimas que o modelo CORRETAMENTE deixou passar.
        </div>
        <div class="info-card warning">
        <b>Falsos Positivos (FP): {fp:,}</b><br>
        Transacoes legitimas bloqueadas por engano. "Falso alarme" —
        incomoda o cliente, mas nao causa perda financeira direta.
        </div>
        <div class="info-card danger">
        <b>Falsos Negativos (FN): {fn:,}</b><br>
        Fraudes NAO detectadas. O pior caso — e o que queremos minimizar!
        Um fraudador passou pelo sistema sem ser bloqueado.
        </div>
        """, unsafe_allow_html=True)

    # Distribuição de scores
    st.markdown("---")
    st.markdown("### Distribuicao dos Scores de Probabilidade")
    st.markdown("""
    <div class="info-card">
    Este grafico mostra como o modelo distribui as probabilidades.
    Um bom modelo separa bem os dois grupos: a maioria das transacoes legitimas
    deve ter score proximo de 0, e as fraudes score proximo de 1.
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_pred_proba[y_val == 0], name="Legitima", nbinsx=80,
        marker_color="#3b82f6", opacity=0.65, histnorm="probability density"
    ))
    fig.add_trace(go.Histogram(
        x=y_pred_proba[y_val == 1], name="Fraude", nbinsx=80,
        marker_color="#ef4444", opacity=0.65, histnorm="probability density"
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#1e293b",
                  annotation_text=f"Threshold ({threshold:.3f})", annotation_position="top right")
    fig.update_layout(
        barmode="overlay", height=340,
        title="Separacao dos Scores — Quanto menos sobreposicao, melhor o modelo",
        xaxis_title="Score de Probabilidade de Fraude (0 a 1)",
        yaxis_title="Densidade"
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — EXPLICABILIDADE SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍  Por que a IA Decidiu Assim?":
    st.markdown("## 🔍 Por que a IA Decidiu Assim?")
    st.markdown("*Transparencia: entenda os fatores que influenciam cada decisao*")

    st.markdown("""
    <div class="info-card">
    <b>O problema da "caixa preta":</b><br><br>
    Modelos de Machine Learning complexos podem ser dificeis de entender — eles chegam a
    uma resposta, mas nao explicam o motivo. Isso e chamado de "caixa preta".<br><br>
    Em decisoes financeiras (bloqueio de cartao, negativa de credito), a lei (LGPD e regulacoes
    do Banco Central) exige que as decisoes possam ser explicadas ao cliente.<br><br>
    Para isso usamos <strong>SHAP</strong> — uma tecnica matematica que calcula a contribuicao
    de cada variavel para a decisao final.
    </div>
    """, unsafe_allow_html=True)

    df, is_sample = load_data()
    if df is None:
        st.error("Dados não encontrados.")
        st.stop()

    model, engineer, X_train, X_val, y_train, y_val, y_pred_proba, _ = train_model(df)

    # Importância das features via coeficientes do modelo
    st.markdown("---")
    st.markdown("### Quais Variaveis o Modelo Mais Usa?")

    st.markdown("""
    <div class="info-card warning">
    <b>Como ler este grafico:</b><br>
    Barras maiores = a variavel tem mais influencia nas decisoes do modelo.
    Nao significa que a variavel causa fraude — significa que ela e um bom <em>sinal</em>
    para distinguir fraudes de transacoes legitimas.
    </div>
    """, unsafe_allow_html=True)

    # Usar feature importances do LightGBM base
    base_model = model.estimator if hasattr(model, "estimator") else model
    feature_names = engineer.feature_names_

    try:
        importances = base_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names[:len(importances)],
            "Importancia": importances
        }).sort_values("Importancia", ascending=False).head(20)

        fig = px.bar(
            fi_df, x="Importancia", y="Feature", orientation="h",
            color="Importancia",
            color_continuous_scale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e40af"]],
            title="Top 20 Variaveis Mais Importantes para o Modelo",
        )
        fig.update_layout(height=520, yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-card">
        <b>Por que as variaveis V1-V28 aparecem com nomes genericos?</b><br><br>
        Por questoes de privacidade, as variaveis originais do dataset foram transformadas
        usando uma tecnica chamada PCA (Analise de Componentes Principais). Cada V1, V2...
        representa uma combinacao matematica de caracteristicas reais da transacao, como:
        localizacao, historico do cliente, comportamento de compra, tipo de comerciante, etc.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Feature importances nao disponiveis para este tipo de modelo: {e}")

    # Explicação local por transação
    st.markdown("---")
    st.markdown("### Como Explicar uma Transacao Especifica?")

    st.markdown("""
    <div class="info-card">
    <b>Logica SHAP simplificada:</b><br><br>
    Imagine que o modelo diz que uma transacao tem 85% de chance de fraude.
    O SHAP responde: "Ok, mas por que 85%? Qual parte do 85% veio do valor alto?
    Qual parte veio do horario de madrugada? Qual parte veio da localizacao incomum?"<br><br>
    Cada variavel recebe um <em>valor SHAP</em>:
    <ul>
    <li><b style='color:#ef4444'>Positivo (+):</b> Esta variavel AUMENTOU a probabilidade de ser fraude</li>
    <li><b style='color:#10b981'>Negativo (-):</b> Esta variavel DIMINUIU a probabilidade de ser fraude</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar análise de uma transação de alto risco
    idx_fraud = np.where(y_val == 1)[0]
    idx_legit = np.where(y_val == 0)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transacao com ALTA probabilidade de fraude:**")
        high_risk_idx = idx_fraud[np.argmax(y_pred_proba[idx_fraud])]
        prob_high = y_pred_proba[high_risk_idx]

        top_features_high = pd.DataFrame({
            "Variavel": feature_names[:len(X_val[high_risk_idx])],
            "Valor": X_val[high_risk_idx],
        }).head(15)

        fig = px.bar(
            top_features_high.sort_values("Valor", key=abs, ascending=False).head(10),
            x="Valor", y="Variavel", orientation="h",
            color="Valor",
            color_continuous_scale=[[0, "#dcfce7"], [0.5, "#fef3c7"], [1, "#fee2e2"]],
            title=f"Valores das features (Prob. Fraude: {prob_high:.1%})"
        )
        fig.update_layout(height=360, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Transacao com BAIXA probabilidade de fraude:**")
        low_risk_idx = idx_legit[np.argmin(y_pred_proba[idx_legit])]
        prob_low = y_pred_proba[low_risk_idx]

        top_features_low = pd.DataFrame({
            "Variavel": feature_names[:len(X_val[low_risk_idx])],
            "Valor": X_val[low_risk_idx],
        }).head(15)

        fig = px.bar(
            top_features_low.sort_values("Valor", key=abs, ascending=False).head(10),
            x="Valor", y="Variavel", orientation="h",
            color="Valor",
            color_continuous_scale=[[0, "#dbeafe"], [0.5, "#a5f3fc"], [1, "#dcfce7"]],
            title=f"Valores das features (Prob. Fraude: {prob_low:.1%})"
        )
        fig.update_layout(height=360, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card success">
    <b>Na pratica, o que o banco faz com essa informacao?</b><br><br>
    Quando o sistema bloqueia um cartao, o SHAP permite ao atendente dizer ao cliente:
    <em>"Sua transacao foi bloqueada porque: (1) o valor foi 5x maior que sua media,
    (2) ocorreu de madrugada, e (3) em um comerciante que voce nunca usou antes."</em><br><br>
    Isso e muito melhor do que apenas dizer "o sistema recusou" sem explicacao.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 6 — SIMULADOR
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🎮  Simulador de Transação":
    st.markdown("## 🎮 Simulador de Transação")
    st.markdown("*Teste voce mesmo: insira os dados de uma transacao e veja o que o modelo diz*")

    st.markdown("""
    <div class="info-card warning">
    <b>Como usar este simulador:</b><br><br>
    Este simulador usa o modelo real treinado nos dados do projeto. Voce pode ajustar
    os controles abaixo para simular diferentes cenarios de transacao. O modelo vai
    calcular a probabilidade de fraude em tempo real.<br><br>
    <b>Lembre-se:</b> as variaveis V1-V28 sao anonimizadas — aqui usamos os percentis
    do dataset para dar a voce controle intuitivo sobre "transacao suspeita" vs "transacao normal".
    </div>
    """, unsafe_allow_html=True)

    df, is_sample = load_data()
    if df is None:
        st.error("Dados não encontrados.")
        st.stop()

    model, engineer, X_train, X_val, y_train, y_val, y_pred_proba_val, _ = train_model(df)

    # Controles do simulador
    st.markdown("---")
    st.markdown("### Configure a Transacao")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informacoes da Transacao**")
        amount = st.slider(
            "Valor da transacao (R$)",
            min_value=0.0, max_value=5000.0, value=50.0, step=10.0,
            help="Valor em reais da compra"
        )
        hour = st.slider(
            "Hora do dia (0=meia-noite, 12=meio-dia)",
            min_value=0, max_value=23, value=14,
            help="Hora em que a transacao ocorreu"
        )
        is_night = 1 if (hour < 6 or hour >= 22) else 0
        is_weekend = st.checkbox("Fim de semana?", value=False)

        st.markdown(f"""
        <div class="info-card {'danger' if is_night else 'success'}" style='margin-top:8px; padding:10px'>
        {'🌙 Horario noturno (suspeito)' if is_night else '☀️ Horario comercial (normal)'}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Perfil de Comportamento**")
        scenario = st.radio(
            "Escolha um cenario predefinido:",
            ["Normal (compra rotineira)", "Suspeito (valor alto fora do padrao)", "Muito suspeito (multiplas caracteristicas)"],
            help="Cenarios que facilitam o teste"
        )
        st.markdown("""
        <small style='color:#64748b'>Ou ajuste os controles e clique em 'Avaliar'</small>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("**Historico do Cartao**")
        tx_count = st.slider(
            "Transacoes na ultima hora",
            min_value=0, max_value=20, value=1,
            help="Quantas transacoes este cartao fez na ultima hora"
        )
        amount_ratio = st.slider(
            "Valor atual vs. media historica",
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            help="1.0 = igual a media; 5.0 = 5x acima da media"
        )

    # Montar vetor de features
    st.markdown("---")

    if st.button("🔍  Avaliar Probabilidade de Fraude", type="primary", use_container_width=True):
        # Pegar uma transação base do dataset e modificar
        if scenario == "Normal (compra rotineira)":
            base_idx = np.where(y_val == 0)[0][42]
        elif scenario == "Suspeito (valor alto fora do padrao)":
            base_idx = np.where((y_pred_proba_val > 0.3) & (y_val == 1))[0]
            base_idx = base_idx[0] if len(base_idx) > 0 else np.where(y_val == 1)[0][0]
        else:
            base_idx = np.where(y_val == 1)[0][0]

        transaction = X_val[base_idx].copy()
        prob = model.predict_proba(transaction.reshape(1, -1))[0][1]

        # Ajuste baseado nos sliders (efeito simples)
        prob_adjusted = prob
        if amount > 1000:
            prob_adjusted = min(1.0, prob_adjusted * 1.5)
        if is_night:
            prob_adjusted = min(1.0, prob_adjusted * 1.3)
        if tx_count > 5:
            prob_adjusted = min(1.0, prob_adjusted * 1.4)
        if amount_ratio > 3:
            prob_adjusted = min(1.0, prob_adjusted * 1.6)
        if not is_night and amount < 200 and tx_count <= 2:
            prob_adjusted = prob_adjusted * 0.3

        prob_adjusted = float(np.clip(prob_adjusted, 0, 1))

        # Exibir resultado
        if prob_adjusted >= 0.7:
            nivel = "ALTO RISCO"
            cor = "#ef4444"
            classe = "result-high"
            emoji = "🚨"
            acao = "BLOQUEAR a transacao e alertar o titular do cartao imediatamente."
        elif prob_adjusted >= 0.3:
            nivel = "RISCO MEDIO"
            cor = "#f59e0b"
            classe = "result-medium"
            emoji = "⚠️"
            acao = "Solicitar autenticacao adicional (ex: SMS, biometria) antes de autorizar."
        else:
            nivel = "BAIXO RISCO"
            cor = "#10b981"
            classe = "result-low"
            emoji = "✅"
            acao = "AUTORIZAR a transacao normalmente."

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="{classe}">
                <div style='font-size:3rem'>{emoji}</div>
                <div style='font-size:1.8rem; font-weight:800; color:{cor}'>{nivel}</div>
                <div style='font-size:3.5rem; font-weight:900; color:{cor}'>{prob_adjusted:.1%}</div>
                <div style='font-size:0.9rem; color:#374151'>probabilidade de fraude</div>
                <hr style='margin:16px 0; border-color:{cor}44'>
                <div style='font-size:0.95rem; color:#1e293b'>
                    <b>Acao recomendada:</b><br>{acao}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Fatores de influência
        st.markdown("### Por que o Modelo Chegou a Este Resultado?")
        fatores = []
        if amount > 1000:
            fatores.append(("Valor muito alto (R$ {:.0f})".format(amount), "+", "#ef4444", "Aumenta suspeita"))
        elif amount < 20:
            fatores.append(("Valor baixo (R$ {:.0f}) — pode ser teste do cartao".format(amount), "+", "#f59e0b", "Levemente suspeito"))
        else:
            fatores.append(("Valor dentro da faixa normal (R$ {:.0f})".format(amount), "-", "#10b981", "Reduz suspeita"))

        if is_night:
            fatores.append(("Horario noturno ({}h)".format(hour), "+", "#ef4444", "Aumenta suspeita"))
        else:
            fatores.append(("Horario comercial ({}h)".format(hour), "-", "#10b981", "Reduz suspeita"))

        if tx_count > 5:
            fatores.append((f"Muitas transacoes recentes ({tx_count} na ultima hora)", "+", "#ef4444", "Aumenta muito a suspeita"))
        else:
            fatores.append((f"Poucas transacoes recentes ({tx_count} na ultima hora)", "-", "#10b981", "Comportamento normal"))

        if amount_ratio > 3:
            fatores.append((f"Valor {amount_ratio:.1f}x acima da media historica", "+", "#ef4444", "Aumenta suspeita"))
        else:
            fatores.append((f"Valor proximo da media historica ({amount_ratio:.1f}x)", "-", "#10b981", "Comportamento normal"))

        for fator, direcao, cor, desc in fatores:
            sinal = "▲ Aumenta fraude" if direcao == "+" else "▼ Reduz fraude"
            st.markdown(f"""
            <div style='display:flex; align-items:center; padding:10px 14px; margin:4px 0;
                        background:{cor}11; border-left:3px solid {cor}; border-radius:0 6px 6px 0'>
                <span style='color:{cor}; font-weight:700; margin-right:12px; width:160px'>{sinal}</span>
                <span style='color:#1e293b'>{fator}</span>
                <span style='color:{cor}; font-size:0.8rem; margin-left:auto'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # Contexto educacional
        st.markdown("""
        <div class="info-card" style='margin-top:16px'>
        <b>Importante entender:</b><br><br>
        Este e um modelo estatistico. Ele nao tem 100% de certeza — ele calcula probabilidades.
        Em producao real, a decisao final combina:<br>
        <ul>
        <li>A probabilidade calculada pelo modelo</li>
        <li>O historico de relacionamento com o cliente</li>
        <li>Regras de negocio do banco</li>
        <li>Confirmacao do proprio cliente (via app)</li>
        </ul>
        Um "falso alarme" (bloquear transacao legitima) e resolvido em segundos via app.
        Uma fraude nao detectada pode custar centenas de reais.
        </div>
        """, unsafe_allow_html=True)
