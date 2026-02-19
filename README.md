# Pipeline de dissertação de mestrado para previsão de churn

Este repositório implementa uma pipeline completa em Python para executar a metodologia descrita na dissertação:

- engenharia de atributos orientada a negócio;
- transformação com `ColumnTransformer` + `OneHotEncoder`;
- balanceamento com `SMOTE` aplicado **somente no treino** dentro da validação cruzada;
- comparação entre 5 algoritmos (`LogisticRegression`, `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`);
- avaliação com métricas para classes desbalanceadas (`accuracy`, `precision`, `recall`, `f1`, `AUC-ROC`, `AUC-PR`, `MCC`);
- explicabilidade com SHAP (global) e LIME (local);
- geração de recomendações gerenciais com base nos fatores explicativos.

## Variáveis derivadas implementadas

1. `Average_Monthly_Charge = TotalCharges / (tenure + 1)`
2. `Service_Count = contagem de serviços adicionais`
3. `Contract_Tenure_Ratio = tenure / duração nominal do contrato`
4. `Monthly_Per_Service = MonthlyCharges / Service_Count`
5. `Senior_Multiple_Services = SeniorCitizen x múltiplos serviços`
6. `High_Value_Customer = MonthlyCharges > P75 (treino)`

## Como executar

```bash
python -m pip install -r requirements.txt
python src/churn_dissertation_pipeline.py --data /caminho/Telco-Customer-Churn.csv --output-dir output
```

## Saídas

- `output/model_comparison.json`: ranking dos modelos por F1 no teste + métricas.
- `output/explainability_report.json`: resultados de SHAP e LIME para o melhor modelo.
- `output/managerial_recommendations.md`: recomendações estratégicas automáticas.

## Observações

- A coluna alvo esperada é `Churn` (com valores `Yes`/`No`).
- A coluna `customerID` é removida automaticamente.
- Quando SHAP/LIME não estiverem disponíveis no ambiente, o erro é registrado no relatório sem interromper a execução.
