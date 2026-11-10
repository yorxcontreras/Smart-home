# Imóvel Inteligente

Aplicação web em Python para análise exploratória e previsão de preços de imóveis. O projeto foi desenvolvido para o trabalho final da disciplina **Tópicos Especiais em Software**.

## Objetivo

Permitir que o usuário envie uma base de imóveis em CSV, JSON ou XML, explore os dados em gráficos interativos, compare algoritmos de Machine Learning e estime o preço de um novo imóvel. Quando uma nova base é enviada, o modelo pode ser treinado novamente pela própria interface.

## Funcionalidades

- Upload de arquivos CSV, JSON e XML.
- Normalização dos nomes das colunas, remoção de duplicidades e tratamento de valores ausentes ou inválidos.
- Base fictícia pronta para demonstração.
- Filtros por cidade e indicadores de preço, área e preço por metro quadrado.
- Gráficos de barras, histograma, dispersão, mapa interativo e matriz de correlação.
- Modelos de Regressão Linear, Árvore de Decisão, Random Forest e KNN.
- Configuração de hiperparâmetros pela interface.
- Comparação por MAE, RMSE e R².
- Formulário para prever o preço de um novo imóvel.
- Botão de treinamento e retreinamento dinâmico.

## Tecnologias

- Python
- Streamlit
- pandas e NumPy
- Plotly
- scikit-learn
- pytest

## Como executar

Recomenda-se Python 3.10 ou superior.

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Como alternativa, dê dois cliques em `executar.bat`. Na primeira execução, o arquivo cria o ambiente virtual e instala as dependências automaticamente.

No Linux ou macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Como testar

```bash
pytest -q
```

## Formato da base

Cada arquivo deve possuir ao menos 20 registros válidos e as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `cidade` | Cidade do imóvel |
| `bairro` | Bairro do imóvel |
| `area_m2` | Área privativa em metros quadrados |
| `quartos` | Quantidade de quartos |
| `banheiros` | Quantidade de banheiros |
| `vagas` | Quantidade de vagas |
| `ano_construcao` | Ano de construção |
| `distancia_metro_km` | Distância até o metrô em quilômetros |
| `latitude` | Latitude usada no mapa |
| `longitude` | Longitude usada no mapa |
| `preco` | Variável-alvo: preço anunciado em reais |

Também são reconhecidos alguns nomes alternativos, como `valor` para `preco`, `metragem` para `area_m2`, `lat` para `latitude` e `lon` para `longitude`.

## Estrutura

```text
imovel-inteligente/
├── app.py
├── data/
│   ├── generate_sample_data.py
│   └── imoveis_exemplo.csv
├── docs/
│   ├── DIVISAO_TAREFAS.md
│   ├── RESULTADOS_BASE_EXEMPLO.md
│   └── ROTEIRO_APRESENTACAO.md
├── src/
│   ├── data_loader.py
│   └── modeling.py
├── tests/
│   ├── test_data_loader.py
│   └── test_modeling.py
├── requirements.txt
├── executar.bat
└── README.md
```

## Metodologia de Machine Learning

O problema é de **regressão**, pois a variável-alvo (`preco`) é numérica. A base é separada de forma reproduzível em 80% para treinamento e 20% para teste. O pré-processamento é aprendido apenas com o conjunto de treinamento para evitar vazamento de dados:

- variáveis numéricas recebem imputação pela mediana e padronização;
- cidade e bairro recebem imputação pelo valor mais frequente e codificação one-hot;
- o modelo é avaliado em dados que não foram usados no treinamento.

Métricas utilizadas:

- **MAE:** erro absoluto médio em reais;
- **RMSE:** penaliza erros maiores e também é expresso em reais;
- **R²:** proporção da variação do preço explicada pelo modelo.

## Observação sobre os dados

A base incluída é **fictícia**, criada somente para demonstrar o funcionamento da aplicação. As previsões não devem ser utilizadas para decisões financeiras reais.

Os resultados reproduzíveis obtidos com essa base estão documentados em `docs/RESULTADOS_BASE_EXEMPLO.md`.

## Tema para enviar ao professor

> Aplicação web para análise exploratória e previsão de preços de imóveis utilizando diferentes modelos de Machine Learning configuráveis pelo usuário.


