import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import joblib

def main():
    dataset_path = "gear_undercutting_ML_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 1. Load Dataset
    df = pd.read_csv(dataset_path)

    # 2. Exclude identifiers, targets, and leakage columns
    # - gear_id: Database key / row identifier
    # - undercut: Primary binary target (0 / 1)
    # - undercut_severity: Target continuous leakage
    # - severity_class: Target multiclass leakage
    # - undercut_margin: Target formula leakage (margin < 0 identically defines undercut)
    leakage_and_id_cols = ['gear_id', 'undercut', 'undercut_severity', 'severity_class', 'undercut_margin']
    feature_cols = [c for c in df.columns if c not in leakage_and_id_cols]

    X = df[feature_cols]
    y = df['undercut']

    categorical_cols = ['material']
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    # 3. Train/Test Stratified Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("=" * 65)
    print("STEP 1: CANDIDATE MODEL BENCHMARK / COMPARISON")
    print("=" * 65)

    candidate_models = {
        'Decision Tree (class_weight="balanced")': DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        'HistGradientBoosting (class_weight="balanced")': HistGradientBoostingClassifier(
            class_weight='balanced', random_state=42
        ),
        'Random Forest (class_weight="balanced")': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        ),
        'Logistic Regression (class_weight="balanced")': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        )
    }

    comparison_results = []
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        comparison_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        trained_pipelines[name] = (pipe, y_pred, y_proba)

    comp_df = pd.DataFrame(comparison_results).sort_values(by='F1-Score', ascending=False)
    print(comp_df.to_string(index=False))

    # Identify the best performing model based on F1-Score and ROC-AUC
    best_model_name = comp_df.iloc[0]['Model']
    print("\n" + "=" * 65)
    print(f"STEP 2: FINAL BEST MODEL EVALUATION -> {best_model_name}")
    print("=" * 65)

    final_pipeline, best_y_pred, best_y_proba = trained_pipelines[best_model_name]

    acc = accuracy_score(y_test, best_y_pred)
    prec = precision_score(y_test, best_y_pred)
    rec = recall_score(y_test, best_y_pred)
    f1 = f1_score(y_test, best_y_pred)
    auc = roc_auc_score(y_test, best_y_proba)
    cm = confusion_matrix(y_test, best_y_pred)

    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted: 0   Predicted: 1")
    print(f"Actual 0 (No Undercut):   {cm[0, 0]:<14} {cm[0, 1]}")
    print(f"Actual 1 (Undercut):      {cm[1, 0]:<14} {cm[1, 1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, best_y_pred, target_names=['No Undercut (0)', 'Undercut (1)'], digits=4))

    # Save final model
    final_model_filename = "gear_undercut_final_model.joblib"
    joblib.dump(final_pipeline, final_model_filename)
    print(f"Final trained model successfully saved to: {final_model_filename}")

    # Also save Decision Tree specifically in case referenced
    dt_pipeline = trained_pipelines['Decision Tree (class_weight="balanced")'][0]
    joblib.dump(dt_pipeline, "gear_undercut_decision_tree_model.joblib")
    print("Decision Tree model saved to: gear_undercut_decision_tree_model.joblib")

if __name__ == "__main__":
    main()
