import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

    # 1. Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Identify and exclude identifier, target, and leakage columns
    # - gear_id: Arbitrary primary key (identifier)
    # - undercut: Primary binary classification target (0 = No Undercut, 1 = Undercut)
    # - undercut_severity: Target leakage (exact continuous undercut severity magnitude)
    # - severity_class: Target leakage (exact multiclass label of undercut severity)
    # - undercut_margin: Target leakage (mathematically x - x_min, perfectly separates undercut at margin < 0)
    leakage_and_id_cols = ['gear_id', 'undercut', 'undercut_severity', 'severity_class', 'undercut_margin']
    feature_cols = [col for col in df.columns if col not in leakage_and_id_cols]

    print("\n--- Feature Selection ---")
    print(f"Target column: undercut (0 = No Undercut, 1 = Undercut)")
    print(f"Excluded identifier & leakage columns: {leakage_and_id_cols}")
    print(f"Input features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df['undercut']

    # 3. Identify categorical and numeric columns
    categorical_cols = ['material']
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # 4. Preprocessing and Model Pipeline
    # Encoding categorical variables strictly within pipeline to avoid data leakage across splits
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    clf = DecisionTreeClassifier(
        criterion='gini',
        class_weight='balanced',  # Appropriately handles class imbalance
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # 5. Train/Test Split (80% train, 20% test, stratified by binary target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\n--- Sample Distribution ---")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"  - Class 0 (No Undercut): {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    print(f"  - Class 1 (Undercut):    {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")
    print(f"Testing samples:  {len(X_test)} (20%)")
    print(f"  - Class 0 (No Undercut): {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")
    print(f"  - Class 1 (Undercut):    {(y_test == 1).sum()} ({(y_test == 1).mean():.2%})")

    # 6. Fit Model Pipeline on Training Data Only
    pipeline.fit(X_train, y_train)

    # 7. Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted: 0   Predicted: 1")
    print(f"Actual 0 (No Undercut):   {cm[0, 0]:<14} {cm[0, 1]}")
    print(f"Actual 1 (Undercut):      {cm[1, 0]:<14} {cm[1, 1]}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Undercut (0)', 'Undercut (1)'], digits=4))

    # 8. Save Trained Model
    model_filename = "gear_undercut_decision_tree_model.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Trained model saved to: {model_filename}")

if __name__ == "__main__":
    main()
