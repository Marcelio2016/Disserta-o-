# ==============================
# SEÇÃO 1 - IMPORTAÇÕES E TIPOS
# ==============================

# Permite usar anotações de tipo modernas sem avaliar imediatamente (melhora compatibilidade).
from __future__ import annotations

# Biblioteca para criar interface de linha de comando (CLI).
import argparse
# Biblioteca para serializar e salvar resultados em JSON.
import json
# Dataclass para estruturar resultados de avaliação de forma organizada.
from dataclasses import dataclass
# Classe Path para manipular caminhos de arquivos de forma robusta.
from pathlib import Path
# Ferramentas padrão para download de arquivo via URL (Google Drive).
from urllib.request import urlretrieve
# Tipo Any para funções que recebem objetos diversos.
from typing import Any

# NumPy para operações numéricas vetorizadas.
import numpy as np
# Pandas para manipulação de tabelas (DataFrames).
import pandas as pd

# SMOTE para oversampling da classe minoritária (churn).
from imblearn.over_sampling import SMOTE
# Pipeline do imbalanced-learn, necessário para encaixar SMOTE no fluxo de treino.
from imblearn.pipeline import Pipeline as ImbPipeline

# Classes-base para criar transformador customizado compatível com scikit-learn.
from sklearn.base import BaseEstimator, TransformerMixin
# ColumnTransformer para aplicar transformações distintas em colunas numéricas/categóricas.
from sklearn.compose import ColumnTransformer
# Modelo Random Forest para classificação.
from sklearn.ensemble import RandomForestClassifier
# Imputador para valores ausentes.
from sklearn.impute import SimpleImputer
# Regressão logística (baseline interpretável).
from sklearn.linear_model import LogisticRegression
# Métricas de classificação e classes desbalanceadas.
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
# Split treino/teste + validação cruzada estratificada + avaliação CV.
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
# Pipeline padrão do sklearn (usado no pré-processamento).
from sklearn.pipeline import Pipeline
# OneHotEncoder para categóricas e StandardScaler para escala numérica.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LightGBM (modelo gradient boosting eficiente).
from lightgbm import LGBMClassifier
# XGBoost (modelo gradient boosting popular em tabulares).
from xgboost import XGBClassifier
# CatBoost (modelo boosting robusto para variáveis categóricas/tabulares).
from catboost import CatBoostClassifier


# ======================================
# SEÇÃO 2 - CONSTANTES E CONFIGURAÇÕES
# ======================================

# Nome da variável-alvo esperada no dataset.
TARGET_COLUMN = "Churn"
# ID do arquivo da base no Google Drive informado pelo usuário.
DEFAULT_GDRIVE_FILE_ID = "1prLfR_9W5WDeCg7KFWj7ZJrbzZMWOkV-"
# URL direta de download do Google Drive construída com o file_id.
DEFAULT_GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DEFAULT_GDRIVE_FILE_ID}"
# Caminho padrão local para salvar a base quando o usuário não passa --data.
DEFAULT_LOCAL_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"


# =================================================
# SEÇÃO 3 - ENGENHARIA DE ATRIBUTOS (FEATURES)
# =================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria variáveis derivadas da metodologia de churn baseada em negócio."""

    # Método construtor da classe.
    def __init__(self) -> None:
        # Guarda o limiar do percentil 75 de MonthlyCharges calculado no treino.
        self.high_value_threshold_: float | None = None

    # Método de ajuste: aprende parâmetros com o treino.
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        # Faz preparação inicial (tipos e imputação mínima em TotalCharges).
        df = self._prepare(X.copy())
        # Calcula percentil 75 de cobrança mensal para definir cliente de alto valor.
        self.high_value_threshold_ = float(df["MonthlyCharges"].quantile(0.75))
        # Retorna a própria instância para compatibilidade com sklearn.
        return self

    # Método de transformação: cria variáveis derivadas em qualquer conjunto (treino/teste).
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Garante que o fit foi executado antes, pois usa limiar aprendido.
        if self.high_value_threshold_ is None:
            raise RuntimeError("FeatureEngineer precisa ser ajustado antes do transform.")

        # Copia e prepara os dados para evitar efeitos colaterais.
        df = self._prepare(X.copy())

        # Lista de colunas de serviços adicionais relevantes para calcular Service_Count.
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        # Mantém apenas colunas que realmente existem no dataset recebido.
        available_service_cols = [c for c in service_cols if c in df.columns]

        # Função auxiliar para converter valores de serviço em indicador binário (0/1).
        def service_to_binary(v: Any) -> int:
            # Se estiver ausente (NaN), considera como não adesão.
            if pd.isna(v):
                return 0
            # Normaliza o texto para comparação consistente.
            s = str(v).strip().lower()
            # Valores que representam adesão ativa ao serviço.
            if s in {"yes", "dsl", "fiber optic", "fiber", "sim", "1", "true"}:
                return 1
            # Demais casos são tratados como não adesão.
            return 0

        # Se há colunas de serviço disponíveis, calcula contagem por cliente.
        if available_service_cols:
            # Aplica conversão binária em cada coluna de serviço.
            service_flags = df[available_service_cols].applymap(service_to_binary)
            # Soma horizontalmente para obter a quantidade total de serviços.
            df["Service_Count"] = service_flags.sum(axis=1)
        else:
            # Se não houver colunas de serviço, define contagem como 0.
            df["Service_Count"] = 0

        # Mapa de duração nominal do contrato em meses.
        contract_month_map = {"month-to-month": 1, "one year": 12, "two year": 24}
        # Converte tipo de contrato para duração nominal (fallback para 1 mês).
        contract_duration = (
            df["Contract"].astype(str).str.lower().map(contract_month_map).fillna(1)
            if "Contract" in df.columns
            else 1
        )

        # Variável 1: gasto médio mensal histórico com ajuste para tenure zero.
        df["Average_Monthly_Charge"] = df["TotalCharges"] / (df["tenure"] + 1)
        # Variável 3: estágio no ciclo contratual.
        df["Contract_Tenure_Ratio"] = df["tenure"] / contract_duration

        # Evita divisão por zero ao calcular custo mensal por serviço.
        service_count_safe = df["Service_Count"].replace(0, np.nan)
        # Variável 4: gasto mensal médio por serviço contratado.
        df["Monthly_Per_Service"] = (
            df["MonthlyCharges"] / service_count_safe
        ).fillna(df["MonthlyCharges"])

        # Recupera indicador de idoso (ou 0 se a coluna não existir).
        senior = df["SeniorCitizen"] if "SeniorCitizen" in df.columns else 0
        # Variável 5: interação idoso + múltiplos serviços (>= 3).
        df["Senior_Multiple_Services"] = ((senior == 1) & (df["Service_Count"] >= 3)).astype(int)
        # Variável 6: cliente de alto valor com base no P75 do treino.
        df["High_Value_Customer"] = (
            df["MonthlyCharges"] > self.high_value_threshold_
        ).astype(int)

        # Retorna DataFrame enriquecido.
        return df

    # Método estático de preparação básica de tipos e valores ausentes.
    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        # Se existir TotalCharges, converte para numérico e trata ausências.
        if "TotalCharges" in df.columns:
            # Converte texto para número; inválidos viram NaN.
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Imputa NaN com mediana para robustez contra outliers.
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # Colunas que devem estar em formato numérico para cálculos.
        for c in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
            # Só converte se a coluna existir no dataset.
            if c in df.columns:
                # Conversão robusta: valores inválidos viram NaN.
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Retorna DataFrame preparado.
        return df


# ========================================
# SEÇÃO 4 - ESTRUTURA DE RESULTADOS
# ========================================

@dataclass
class EvaluationResult:
    # Nome do modelo avaliado.
    model_name: str
    # Métricas médias da validação cruzada.
    cv_metrics: dict[str, float]
    # Métricas no conjunto de teste hold-out.
    test_metrics: dict[str, float]


# ============================================
# SEÇÃO 5 - CARGA E PRÉ-PROCESSAMENTO DE DADOS
# ============================================

# Resolve caminho da base: usa --data quando informado ou baixa automaticamente do Google Drive.
def resolve_data_path(data_arg: str | None) -> Path:
    # Se usuário passou --data, usa esse caminho diretamente.
    if data_arg:
        return Path(data_arg)

    # Se o arquivo padrão já existir localmente, reaproveita sem novo download.
    if DEFAULT_LOCAL_DATA_PATH.exists():
        return DEFAULT_LOCAL_DATA_PATH

    # Garante que a pasta de destino exista antes de baixar.
    DEFAULT_LOCAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Faz download da base a partir do Google Drive para o caminho padrão local.
    urlretrieve(DEFAULT_GDRIVE_DOWNLOAD_URL, DEFAULT_LOCAL_DATA_PATH)
    # Retorna caminho local pronto para leitura do CSV.
    return DEFAULT_LOCAL_DATA_PATH


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    # Lê CSV informado pelo usuário no argumento --data.
    df = pd.read_csv(path)
    # Verifica existência da coluna alvo esperada.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada no dataset.")

    # Normaliza alvo para minúsculas e mapeia Yes/No para 1/0.
    y = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    # Garante que não há valores fora do padrão Yes/No.
    if y.isna().any():
        raise ValueError("A coluna Churn deve conter valores Yes/No.")

    # Separa variáveis explicativas removendo alvo.
    X = df.drop(columns=[TARGET_COLUMN])
    # Remove identificador de cliente para evitar vazamento e ruído.
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    # Retorna matriz X e vetor y.
    return X, y.astype(int)


# Monta o pré-processador com fluxo separado para numéricos e categóricos.
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    # Identifica colunas categóricas por tipo.
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Define numéricas como complemento das categóricas.
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Pipeline numérico: imputação de mediana + padronização.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    # Pipeline categórico: one-hot encoding com tolerância a categorias novas.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combina os dois fluxos em um único transformador de colunas.
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


# ==================================
# SEÇÃO 6 - MODELOS E MÉTRICAS
# ==================================

def build_models(random_state: int = 42) -> dict[str, Any]:
    # Retorna dicionário com os 5 modelos definidos na metodologia.
    return {
        # Baseline linear interpretável.
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        # Ensemble bagging robusto a ruído.
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
        ),
        # Gradient boosting via XGBoost.
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        # Gradient boosting via LightGBM.
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, random_state=random_state, n_jobs=-1, verbose=-1
        ),
        # Gradient boosting via CatBoost.
        "CatBoost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            verbose=False,
        ),
    }


# Calcula conjunto padronizado de métricas para y verdadeiro, predição e probabilidade.
def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    # Retorna dicionário com todas as métricas requeridas para classe desbalanceada.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


# =====================================================
# SEÇÃO 7 - TREINO, VALIDAÇÃO CRUZADA E AVALIAÇÃO FINAL
# =====================================================

def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[
    list[EvaluationResult],
    dict[str, ImbPipeline],
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
]:
    # Cria hold-out estratificado para avaliação final imparcial.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Instancia o engenheiro de atributos.
    engineer = FeatureEngineer()
    # Ajusta engenharia no treino e transforma treino para descobrir tipos de colunas.
    X_train_fe = engineer.fit_transform(X_train)
    # Transforma teste com parâmetros aprendidos no treino.
    _ = engineer.transform(X_test)

    # Constrói preprocessor com base na estrutura do treino já enriquecido.
    preprocessor = build_preprocessor(X_train_fe)
    # Cria os modelos candidatos.
    models = build_models(random_state=random_state)

    # Define validação cruzada estratificada com reprodutibilidade.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # Define métricas a serem computadas no CV.
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc",
        "auc_pr": "average_precision",
        "mcc": "matthews_corrcoef",
    }

    # Lista para armazenar resultados de todos os modelos.
    results: list[EvaluationResult] = []
    # Dicionário para armazenar pipelines treinadas, útil para explicabilidade posterior.
    fitted_pipelines: dict[str, ImbPipeline] = {}

    # Itera por cada modelo candidato.
    for name, model in models.items():
        # Monta pipeline completa: engenharia -> pré-processamento -> SMOTE -> modelo.
        pipeline = ImbPipeline(
            steps=[
                ("feature_engineer", FeatureEngineer()),
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

        # Executa validação cruzada no treino sem tocar o conjunto de teste.
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        # Calcula média de cada métrica no CV.
        cv_metrics = {k: float(np.mean(v)) for k, v in cv_result.items() if k.startswith("test_")}

        # Ajusta a pipeline final no treino inteiro.
        pipeline.fit(X_train, y_train)
        # Guarda pipeline treinada para uso posterior (SHAP/LIME).
        fitted_pipelines[name] = pipeline

        # Prediz probabilidade de churn no teste.
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        # Converte probabilidade em classe binária com threshold 0.5.
        y_pred = (y_prob >= 0.5).astype(int)
        # Calcula métricas no teste.
        test_metrics = metric_dict(y_test.values, y_pred, y_prob)

        # Registra resultados estruturados desse modelo.
        results.append(EvaluationResult(name, cv_metrics, test_metrics))

    # Retorna resultados, pipelines treinadas e split usado.
    return results, fitted_pipelines, (X_train, X_test, y_train, y_test)


# ==========================================
# SEÇÃO 8 - EXPLICABILIDADE (SHAP E LIME)
# ==========================================

def generate_explainability(
    best_model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    # Estrutura base do relatório de explicabilidade.
    explainability: dict[str, Any] = {"best_model": best_model_name}

    # Aplica as mesmas transformações do pipeline sem SMOTE, para explicar no espaço final de features.
    transformed_train = pipeline.named_steps["preprocessor"].transform(
        pipeline.named_steps["feature_engineer"].transform(X_train)
    )
    # Transforma também o conjunto de teste.
    transformed_test = pipeline.named_steps["preprocessor"].transform(
        pipeline.named_steps["feature_engineer"].transform(X_test)
    )

    # Se a matriz vier esparsa, converte para densa para facilitar integração com SHAP/LIME.
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()
    # Mesmo processo para teste.
    if hasattr(transformed_test, "toarray"):
        transformed_test = transformed_test.toarray()

    # Recupera nomes finais das features após ColumnTransformer+OHE.
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Bloco SHAP com tratamento de falhas para não quebrar pipeline principal.
    try:
        # Importa SHAP apenas quando necessário.
        import shap

        # Recupera estimador treinado.
        model = pipeline.named_steps["model"]
        # Limita amostra para custo computacional controlado.
        sample = transformed_test[: min(300, len(transformed_test))]

        # Para regressão logística, usa explicador linear.
        if best_model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, transformed_train)
            shap_values = explainer.shap_values(sample)
        else:
            # Para modelos baseados em árvore, usa TreeExplainer.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

        # Trata saída de SHAP quando retorna lista por classe.
        if isinstance(shap_values, list):
            shap_arr = np.abs(shap_values[1]).mean(axis=0)
        else:
            # Caso retorne array direto, calcula média absoluta por feature.
            shap_arr = np.abs(shap_values).mean(axis=0)

        # Ordena features por importância global decrescente e pega top 15.
        top_idx = np.argsort(shap_arr)[::-1][:15]
        # Salva ranking SHAP no relatório.
        explainability["shap_global_top15"] = [
            {"feature": feature_names[i], "mean_abs_shap": float(shap_arr[i])} for i in top_idx
        ]
    except Exception as exc:  # noqa: BLE001
        # Registra erro para transparência sem interromper execução.
        explainability["shap_error"] = str(exc)

    # Bloco LIME com tratamento de falhas.
    try:
        # Importa explicador tabular do LIME.
        from lime.lime_tabular import LimeTabularExplainer

        # Inicializa LIME com dados transformados de treino.
        explainer = LimeTabularExplainer(
            training_data=transformed_train,
            feature_names=list(feature_names),
            class_names=["No Churn", "Churn"],
            mode="classification",
        )
        # Recupera modelo treinado para função predict_proba.
        model = pipeline.named_steps["model"]
        # Explica a primeira observação do teste.
        exp = explainer.explain_instance(
            transformed_test[0],
            model.predict_proba,
            num_features=10,
            top_labels=1,
        )
        # Exporta regras locais e seus pesos.
        explainability["lime_instance_0"] = [
            {"rule": r, "weight": float(w)} for r, w in exp.as_list(label=1)
        ]
    except Exception as exc:  # noqa: BLE001
        # Registra erro de LIME, mantendo execução.
        explainability["lime_error"] = str(exc)

    # Define arquivo final de explicabilidade.
    out_file = output_dir / "explainability_report.json"
    # Salva relatório em JSON legível.
    out_file.write_text(json.dumps(explainability, ensure_ascii=False, indent=2), encoding="utf-8")
    # Retorna estrutura em memória.
    return explainability


# ========================================================
# SEÇÃO 9 - RECOMENDAÇÕES GERENCIAIS A PARTIR DO SHAP
# ========================================================

def create_managerial_recommendations(explainability: dict[str, Any], output_dir: Path) -> None:
    # Inicia linhas do arquivo markdown de recomendações.
    lines = [
        "# Recomendações gerenciais de retenção",
        "",
        "As recomendações abaixo foram geradas a partir das variáveis mais influentes no SHAP global.",
        "",
    ]

    # Recupera top features SHAP (se houver).
    top = explainability.get("shap_global_top15", [])
    # Se não houver SHAP, registra aviso no relatório.
    if not top:
        lines.append("Não foi possível gerar recomendações automáticas porque o SHAP não foi executado com sucesso.")
    else:
        # Gera recomendações focando nas 5 variáveis mais importantes.
        for item in top[:5]:
            # Nome da variável importante.
            feat = item["feature"]
            # Recomendação orientada à ação gerencial.
            lines.append(f"- Priorizar monitoramento e ações de retenção associados ao fator **{feat}**.")

    # Escreve arquivo markdown com recomendações.
    (output_dir / "managerial_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


# ==========================================
# SEÇÃO 10 - ORQUESTRAÇÃO (FUNÇÃO MAIN)
# ==========================================

def main() -> None:
    # Cria parser de argumentos CLI.
    parser = argparse.ArgumentParser(description="Pipeline de dissertação para predição de churn")
    # Argumento obrigatório com caminho do CSV.
    parser.add_argument("--data", required=False, help="Caminho para CSV da base Telco Customer Churn")
    # Argumento opcional para diretório de saída.
    parser.add_argument("--output-dir", default="output", help="Diretório de saída para relatórios")
    # Faz parsing dos argumentos passados pelo usuário.
    args = parser.parse_args()

    # Cria diretório de saída se não existir.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carrega dados de entrada.
    resolved_data_path = resolve_data_path(args.data)
    # Carrega dados de entrada do caminho resolvido (local informado ou Google Drive).
    X, y = load_data(str(resolved_data_path))
    # Treina, valida e compara modelos.
    results, fitted_pipelines, split = evaluate_models(X, y)
    # Recupera partições para etapa de explicabilidade.
    X_train, X_test, _, _ = split

    # Estrutura lista serializável com resultados dos modelos.
    results_payload = []
    # Percorre resultados para converter dataclass em dicionários simples.
    for r in results:
        results_payload.append(
            {
                "model": r.model_name,
                "cv": r.cv_metrics,
                "test": r.test_metrics,
            }
        )

    # Ordena modelos por F1 no teste (maior para menor).
    ranking = sorted(results_payload, key=lambda x: x["test"]["f1"], reverse=True)
    # Identifica nome do melhor modelo.
    best_name = ranking[0]["model"]
    # Gera relatório de explicabilidade para o melhor modelo.
    explainability = generate_explainability(best_name, fitted_pipelines[best_name], X_train, X_test, output_dir)
    # Gera recomendações estratégicas com base em explicabilidade.
    create_managerial_recommendations(explainability, output_dir)

    # Salva ranking completo dos modelos e métricas no JSON final.
    (output_dir / "model_comparison.json").write_text(
        json.dumps({"ranking_by_test_f1": ranking}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Feedback de conclusão para terminal.
    print("Execução concluída.")
    # Mostra o caminho da base efetivamente utilizada na execução.
    print(f"Base utilizada: {resolved_data_path.resolve()}")
    # Informa melhor modelo por F1 de teste.
    print(f"Melhor modelo (F1 teste): {best_name}")
    # Informa caminho absoluto dos relatórios gerados.
    print(f"Relatórios em: {output_dir.resolve()}")


# Ponto de entrada quando o script é executado diretamente.
if __name__ == "__main__":
    # Executa fluxo principal da aplicação.
    main()
