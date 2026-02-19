from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


TARGET_COLUMN = "Churn"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria variáveis derivadas orientadas ao domínio de churn em telecom."""

    def __init__(self) -> None:
        self.high_value_threshold_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        df = self._prepare(X.copy())
        self.high_value_threshold_ = float(df["MonthlyCharges"].quantile(0.75))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.high_value_threshold_ is None:
            raise RuntimeError("FeatureEngineer precisa ser ajustado antes do transform.")

        df = self._prepare(X.copy())

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
        available_service_cols = [c for c in service_cols if c in df.columns]

        def service_to_binary(v: Any) -> int:
            if pd.isna(v):
                return 0
            s = str(v).strip().lower()
            if s in {"yes", "dsl", "fiber optic", "fiber", "sim", "1", "true"}:
                return 1
            return 0

        if available_service_cols:
            service_flags = df[available_service_cols].applymap(service_to_binary)
            df["Service_Count"] = service_flags.sum(axis=1)
        else:
            df["Service_Count"] = 0

        contract_month_map = {"month-to-month": 1, "one year": 12, "two year": 24}
        contract_duration = (
            df["Contract"].astype(str).str.lower().map(contract_month_map).fillna(1)
            if "Contract" in df.columns
            else 1
        )

        df["Average_Monthly_Charge"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["Contract_Tenure_Ratio"] = df["tenure"] / contract_duration

        service_count_safe = df["Service_Count"].replace(0, np.nan)
        df["Monthly_Per_Service"] = (
            df["MonthlyCharges"] / service_count_safe
        ).fillna(df["MonthlyCharges"])

        senior = df["SeniorCitizen"] if "SeniorCitizen" in df.columns else 0
        df["Senior_Multiple_Services"] = ((senior == 1) & (df["Service_Count"] >= 3)).astype(int)
        df["High_Value_Customer"] = (
            df["MonthlyCharges"] > self.high_value_threshold_
        ).astype(int)

        return df

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        for c in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df


@dataclass
class EvaluationResult:
    model_name: str
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada no dataset.")

    y = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("A coluna Churn deve conter valores Yes/No.")

    X = df.drop(columns=[TARGET_COLUMN])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y.astype(int)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
        ),
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
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, random_state=random_state, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            verbose=False,
        ),
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[list[EvaluationResult], dict[str, ImbPipeline], tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    engineer = FeatureEngineer()
    X_train_fe = engineer.fit_transform(X_train)
    X_test_fe = engineer.transform(X_test)

    preprocessor = build_preprocessor(X_train_fe)
    models = build_models(random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc",
        "auc_pr": "average_precision",
        "mcc": "matthews_corrcoef",
    }

    results: list[EvaluationResult] = []
    fitted_pipelines: dict[str, ImbPipeline] = {}

    for name, model in models.items():
        pipeline = ImbPipeline(
            steps=[
                ("feature_engineer", engineer),
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_metrics = {k: float(np.mean(v)) for k, v in cv_result.items() if k.startswith("test_")}

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        test_metrics = metric_dict(y_test.values, y_pred, y_prob)

        results.append(EvaluationResult(name, cv_metrics, test_metrics))

    return results, fitted_pipelines, (X_train, X_test, y_train, y_test)


def generate_explainability(
    best_model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    explainability: dict[str, Any] = {"best_model": best_model_name}

    transformed_train = pipeline.named_steps["preprocessor"].transform(
        pipeline.named_steps["feature_engineer"].transform(X_train)
    )
    transformed_test = pipeline.named_steps["preprocessor"].transform(
        pipeline.named_steps["feature_engineer"].transform(X_test)
    )

    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()
    if hasattr(transformed_test, "toarray"):
        transformed_test = transformed_test.toarray()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    try:
        import shap

        model = pipeline.named_steps["model"]
        sample = transformed_test[: min(300, len(transformed_test))]

        if best_model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, transformed_train)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_arr = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_arr = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(shap_arr)[::-1][:15]
        explainability["shap_global_top15"] = [
            {"feature": feature_names[i], "mean_abs_shap": float(shap_arr[i])} for i in top_idx
        ]
    except Exception as exc:  # noqa: BLE001
        explainability["shap_error"] = str(exc)

    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=transformed_train,
            feature_names=list(feature_names),
            class_names=["No Churn", "Churn"],
            mode="classification",
        )
        model = pipeline.named_steps["model"]
        exp = explainer.explain_instance(
            transformed_test[0],
            model.predict_proba,
            num_features=10,
            top_labels=1,
        )
        explainability["lime_instance_0"] = [
            {"rule": r, "weight": float(w)} for r, w in exp.as_list(label=1)
        ]
    except Exception as exc:  # noqa: BLE001
        explainability["lime_error"] = str(exc)

    out_file = output_dir / "explainability_report.json"
    out_file.write_text(json.dumps(explainability, ensure_ascii=False, indent=2), encoding="utf-8")
    return explainability


def create_managerial_recommendations(explainability: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Recomendações gerenciais de retenção",
        "",
        "As recomendações abaixo foram geradas a partir das variáveis mais influentes no SHAP global.",
        "",
    ]

    top = explainability.get("shap_global_top15", [])
    if not top:
        lines.append("Não foi possível gerar recomendações automáticas porque o SHAP não foi executado com sucesso.")
    else:
        for item in top[:5]:
            feat = item["feature"]
            lines.append(f"- Priorizar monitoramento e ações de retenção associados ao fator **{feat}**.")

    (output_dir / "managerial_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de dissertação para predição de churn")
    parser.add_argument("--data", required=True, help="Caminho para CSV da base Telco Customer Churn")
    parser.add_argument("--output-dir", default="output", help="Diretório de saída para relatórios")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data)
    results, fitted_pipelines, split = evaluate_models(X, y)
    X_train, X_test, _, _ = split

    results_payload = []
    for r in results:
        results_payload.append(
            {
                "model": r.model_name,
                "cv": r.cv_metrics,
                "test": r.test_metrics,
            }
        )

    ranking = sorted(results_payload, key=lambda x: x["test"]["f1"], reverse=True)
    best_name = ranking[0]["model"]
    explainability = generate_explainability(best_name, fitted_pipelines[best_name], X_train, X_test, output_dir)
    create_managerial_recommendations(explainability, output_dir)

    (output_dir / "model_comparison.json").write_text(
        json.dumps({"ranking_by_test_f1": ranking}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Execução concluída.")
    print(f"Melhor modelo (F1 teste): {best_name}")
    print(f"Relatórios em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
