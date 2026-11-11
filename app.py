"""Aplicação Streamlit do projeto Imóvel Inteligente."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    COLUMN_DESCRIPTIONS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    load_dataset,
)
from src.modeling import DEFAULT_PARAMETERS, MODEL_NAMES, compare_models, train_and_evaluate


APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA = APP_DIR / "data" / "imoveis_exemplo.csv"

st.set_page_config(
    page_title="Imóvel Inteligente",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #f4f8f7 0%, #e9f2ef 100%);
            border: 1px solid #dce8e4;
            padding: 1rem;
            border-radius: 0.8rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #18312c;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        .app-subtitle {color: #52645f; margin-top: -0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def brl(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", ".")


def fingerprint(dataframe: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


@st.cache_data(show_spinner=False)
def load_default_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    return load_dataset(SAMPLE_DATA)


def model_parameters(model_name: str) -> dict:
    if model_name == "Regressão Linear":
        st.caption("Esse modelo não possui hiperparâmetros obrigatórios nesta demonstração.")
        return {}
    if model_name == "Árvore de Decisão":
        return {
            "max_depth": st.slider("Profundidade máxima", 2, 30, 8),
            "min_samples_leaf": st.slider("Mínimo de registros por folha", 1, 20, 2),
        }
    if model_name == "Random Forest":
        return {
            "n_estimators": st.slider("Quantidade de árvores", 50, 500, 200, 50),
            "max_depth": st.slider("Profundidade máxima", 2, 30, 12),
            "min_samples_leaf": st.slider("Mínimo de registros por folha", 1, 20, 1),
        }
    return {
        "n_neighbors": st.slider("Número de vizinhos", 2, 25, 7),
        "weights": st.selectbox("Peso dos vizinhos", ["distance", "uniform"]),
    }


st.title("🏠 Imóvel Inteligente")
st.markdown(
    '<p class="app-subtitle">Análise interativa e previsão de preços de imóveis com Machine Learning.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Base de dados")
    uploaded_file = st.file_uploader(
        "Envie uma base com o mesmo esquema",
        type=["csv", "json", "xml"],
        help="A aplicação aceita CSV, JSON e XML com as colunas descritas em Sobre a base.",
    )
    st.caption("Sem upload, a base fictícia incluída no projeto é utilizada.")

try:
    if uploaded_file is None:
        data, quality_report = load_default_dataset()
        data_source = "Base demonstrativa"
    else:
        data, quality_report = load_dataset(uploaded_file, uploaded_file.name)
        data_source = uploaded_file.name
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

data_id = fingerprint(data)
if st.session_state.get("active_data_id") != data_id:
    st.session_state["active_data_id"] = data_id
    st.session_state.pop("trained_result", None)
    st.session_state.pop("comparison", None)

with st.sidebar:
    st.success(f"{data_source}: {len(data)} registros válidos")
    with st.expander("Relatório da limpeza"):
        st.json(quality_report)
    cities = sorted(data["cidade"].unique())
    selected_cities = st.multiselect("Filtrar cidades nos gráficos", cities, default=cities)
    filtered = data[data["cidade"].isin(selected_cities)] if selected_cities else data.iloc[0:0]

overview_tab, charts_tab, ml_tab, prediction_tab, data_tab = st.tabs(
    ["Visão geral", "Análises", "Machine Learning", "Previsão", "Sobre a base"]
)

with overview_tab:
    if filtered.empty:
        st.warning("Selecione ao menos uma cidade na barra lateral.")
    else:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Imóveis", f"{len(filtered):,}".replace(",", "."))
        metric_columns[1].metric("Preço médio", brl(filtered["preco"].mean()))
        metric_columns[2].metric("Área média", f"{filtered['area_m2'].mean():.1f} m²")
        metric_columns[3].metric(
            "Preço médio/m²", brl((filtered["preco"] / filtered["area_m2"]).mean())
        )

        left, right = st.columns(2)
        with left:
            city_summary = (
                filtered.groupby("cidade", as_index=False)
                .agg(preco_medio=("preco", "mean"), quantidade=("preco", "size"))
                .sort_values("preco_medio", ascending=False)
            )
            figure = px.bar(
                city_summary,
                x="cidade",
                y="preco_medio",
                color="quantidade",
                labels={"cidade": "Cidade", "preco_medio": "Preço médio", "quantidade": "Imóveis"},
                title="Preço médio por cidade",
                color_continuous_scale="Teal",
            )
            figure.update_layout(yaxis_tickprefix="R$ ", coloraxis_showscale=False)
            st.plotly_chart(figure, width="stretch")
        with right:
            figure = px.histogram(
                filtered,
                x="preco",
                nbins=30,
                color="cidade",
                labels={"preco": "Preço", "count": "Quantidade", "cidade": "Cidade"},
                title="Distribuição dos preços",
            )
            figure.update_layout(xaxis_tickprefix="R$ ", bargap=0.05)
            st.plotly_chart(figure, width="stretch")

        st.subheader("Amostra dos dados")
        st.dataframe(filtered.head(20), width="stretch", hide_index=True)

with charts_tab:
    if filtered.empty:
        st.warning("Selecione ao menos uma cidade na barra lateral.")
    else:
        first, second = st.columns(2)
        with first:
            neighborhood_summary = (
                filtered.groupby(["cidade", "bairro"], as_index=False)
                .agg(preco_medio=("preco", "mean"), quantidade=("preco", "size"))
                .sort_values("preco_medio", ascending=False)
            )
            figure = px.bar(
                neighborhood_summary,
                x="bairro",
                y="preco_medio",
                color="cidade",
                hover_data=["quantidade"],
                labels={"bairro": "Bairro", "preco_medio": "Preço médio", "cidade": "Cidade"},
                title="Preço médio por bairro",
            )
            figure.update_layout(yaxis_tickprefix="R$ ")
            st.plotly_chart(figure, width="stretch")
        with second:
            figure = px.scatter(
                filtered,
                x="area_m2",
                y="preco",
                color="cidade",
                size="quartos",
                hover_data=["bairro", "banheiros", "vagas"],
                labels={"area_m2": "Área (m²)", "preco": "Preço", "cidade": "Cidade"},
                title="Relação entre área e preço",
            )
            figure.update_layout(yaxis_tickprefix="R$ ")
            st.plotly_chart(figure, width="stretch")

        st.subheader("Mapa interativo")
        map_data = filtered.rename(columns={"latitude": "lat", "longitude": "lon"})
        st.map(map_data[["lat", "lon"]], width="stretch")

        correlation_columns = [
            "area_m2",
            "quartos",
            "banheiros",
            "vagas",
            "ano_construcao",
            "distancia_metro_km",
            "preco",
        ]
        correlation = filtered[correlation_columns].corr(numeric_only=True)
        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlação entre variáveis numéricas",
        )
        st.plotly_chart(heatmap, width="stretch")

with ml_tab:
    st.subheader("Treinamento dinâmico")
    st.write(
        "Escolha o algoritmo, altere os parâmetros e treine novamente. "
        "Os resultados usam 80% dos dados para treino e 20% para teste."
    )
    controls, explanation = st.columns([1, 1])
    with controls:
        selected_model = st.selectbox("Modelo", MODEL_NAMES, index=2)
        parameters = model_parameters(selected_model)
        train_clicked = st.button("Treinar / retreinar modelo", type="primary", width="stretch")
        compare_clicked = st.button("Comparar todos os modelos", width="stretch")
    with explanation:
        st.info(
            "MAE e RMSE medem o erro em reais: quanto menores, melhor. "
            "O R² mede quanto da variação dos preços é explicada pelo modelo: quanto mais perto de 1, melhor."
        )

    if train_clicked:
        with st.spinner("Treinando e avaliando o modelo..."):
            st.session_state["trained_result"] = train_and_evaluate(
                data, selected_model, parameters
            )
            st.session_state["trained_data_id"] = data_id

    if compare_clicked:
        with st.spinner("Comparando modelos..."):
            st.session_state["comparison"] = compare_models(data)

    result = st.session_state.get("trained_result")
    if result is not None and st.session_state.get("trained_data_id") == data_id:
        st.success(f"Modelo treinado: {result.model_name}")
        cols = st.columns(5)
        cols[0].metric("MAE", brl(result.metrics["MAE"]))
        cols[1].metric("RMSE", brl(result.metrics["RMSE"]))
        cols[2].metric("R²", f"{result.metrics['R²']:.3f}")
        cols[3].metric("Treino", result.train_rows)
        cols[4].metric("Teste", result.test_rows)

        predictions = result.predictions
        figure = px.scatter(
            predictions,
            x="valor_real",
            y="valor_previsto",
            labels={"valor_real": "Preço real", "valor_previsto": "Preço previsto"},
            title="Valores reais versus previstos",
        )
        min_value = min(predictions["valor_real"].min(), predictions["valor_previsto"].min())
        max_value = max(predictions["valor_real"].max(), predictions["valor_previsto"].max())
        figure.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line={"dash": "dash", "color": "#dd6b20"},
        )
        figure.update_layout(xaxis_tickprefix="R$ ", yaxis_tickprefix="R$ ")
        st.plotly_chart(figure, width="stretch")
    else:
        st.caption("Treine um modelo para visualizar as métricas e habilitar a previsão.")

    comparison = st.session_state.get("comparison")
    if comparison is not None:
        st.subheader("Comparação dos modelos")
        display_comparison = comparison.copy()
        display_comparison["MAE"] = display_comparison["MAE"].map(brl)
        display_comparison["RMSE"] = display_comparison["RMSE"].map(brl)
        display_comparison["R²"] = display_comparison["R²"].map(lambda value: f"{value:.3f}")
        st.dataframe(display_comparison, width="stretch", hide_index=True)

with prediction_tab:
    st.subheader("Simular o preço de um imóvel")
    result = st.session_state.get("trained_result")
    if result is None or st.session_state.get("trained_data_id") != data_id:
        st.warning("Treine um modelo na aba Machine Learning antes de fazer uma previsão.")
    else:
        with st.form("prediction_form"):
            first, second, third = st.columns(3)
            with first:
                city = st.selectbox("Cidade", sorted(data["cidade"].unique()))
                neighborhood_options = sorted(data.loc[data["cidade"] == city, "bairro"].unique())
                neighborhood = st.selectbox("Bairro", neighborhood_options)
                area = st.number_input("Área (m²)", 20.0, 1000.0, 80.0, 5.0)
            with second:
                bedrooms = st.number_input("Quartos", 0, 15, 2)
                bathrooms = st.number_input("Banheiros", 0, 15, 2)
                parking = st.number_input("Vagas", 0, 10, 1)
            with third:
                year = st.number_input("Ano de construção", 1900, 2030, 2010)
                metro_distance = st.number_input(
                    "Distância do metrô (km)", 0.0, 30.0, 1.0, 0.1
                )
            submitted = st.form_submit_button("Calcular preço estimado", type="primary")

        if submitted:
            input_data = pd.DataFrame(
                [
                    {
                        "cidade": city,
                        "bairro": neighborhood,
                        "area_m2": area,
                        "quartos": bedrooms,
                        "banheiros": bathrooms,
                        "vagas": parking,
                        "ano_construcao": year,
                        "distancia_metro_km": metro_distance,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )
            estimated_price = float(result.pipeline.predict(input_data)[0])
            st.metric("Preço estimado", brl(max(estimated_price, 0)))
            st.caption(
                f"Estimativa gerada pelo modelo {result.model_name}. "
                "O valor é demonstrativo e não substitui uma avaliação profissional."
            )

with data_tab:
    st.subheader("Esquema esperado")
    schema = pd.DataFrame(
        [{"coluna": column, "descrição": description} for column, description in COLUMN_DESCRIPTIONS.items()]
    )
    st.dataframe(schema, width="stretch", hide_index=True)
    st.download_button(
        "Baixar base de exemplo",
        data=SAMPLE_DATA.read_bytes(),
        file_name="imoveis_exemplo.csv",
        mime="text/csv",
    )
    st.markdown(
        "A variável que o modelo tenta prever é `preco`. Latitude e longitude são usadas no mapa; "
        "as demais colunas formam as entradas dos modelos."
    )

