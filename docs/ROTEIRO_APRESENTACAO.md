# Roteiro de apresentação

## 1. Problema e objetivo

- Apresentar o tema: análise e previsão de preços de imóveis.
- Explicar que o usuário pode trocar a base sem alterar o código, desde que mantenha o esquema esperado.
- Informar que a base demonstrativa é fictícia.

## 2. Demonstração do front-end

1. Abrir a aplicação sem upload e mostrar os indicadores.
2. Filtrar uma cidade.
3. Mostrar os gráficos e o mapa.
4. Abrir a aba Machine Learning.
5. Comparar todos os modelos.
6. Escolher um modelo e alterar um hiperparâmetro.
7. Treinar e interpretar MAE, RMSE e R².
8. Simular um imóvel na aba Previsão.
9. Fazer upload de uma nova base e mostrar que o modelo precisa ser retreinado.

## 3. Pontos técnicos para a defesa

- A validação impede treinamento com colunas ausentes ou poucos registros.
- O `ColumnTransformer` trata variáveis numéricas e categóricas de formas diferentes.
- Todo o pré-processamento faz parte do `Pipeline`, evitando vazamento de dados.
- A divisão treino/teste mede o desempenho em registros não vistos.
- A comparação usa a mesma divisão aleatória para todos os modelos.
- A previsão usa exatamente o pipeline que foi treinado.

## 4. Limitações e melhorias futuras

- A qualidade da previsão depende da qualidade e representatividade da base.
- Uma versão futura poderia usar dados reais, validação cruzada e otimização automática de hiperparâmetros.
- Também seria possível salvar modelos treinados, adicionar autenticação e publicar a aplicação na nuvem.

