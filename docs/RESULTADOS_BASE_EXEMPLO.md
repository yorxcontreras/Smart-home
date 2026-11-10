# Resultados da base demonstrativa

Os números abaixo foram obtidos com a base fictícia de 420 imóveis incluída no projeto. Todos os modelos utilizaram a mesma separação reproduzível: 80% dos registros para treino e 20% para teste, com `random_state=42`.

| Posição | Modelo | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
| 1 | Random Forest | R$ 71.672 | R$ 95.388 | 0,942 |
| 2 | Regressão Linear | R$ 78.132 | R$ 100.675 | 0,936 |
| 3 | Árvore de Decisão | R$ 96.762 | R$ 124.611 | 0,901 |
| 4 | KNN | R$ 127.733 | R$ 161.690 | 0,834 |

## Interpretação

Na base demonstrativa, o Random Forest teve o menor erro e o maior R². Seu MAE de aproximadamente R$ 71,7 mil significa que, em média, a diferença absoluta entre o preço real e o previsto foi desse valor. O R² de 0,942 indica que o modelo explicou aproximadamente 94,2% da variação dos preços presentes no conjunto de teste.

Esses resultados servem apenas para demonstrar a aplicação. Como a base é fictícia, eles não representam o mercado imobiliário real e não devem ser usados para decisões financeiras.

