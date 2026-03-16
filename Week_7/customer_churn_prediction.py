import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import shap

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from pytorch_tabnet.callbacks import Callback

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv('Telco-Customer-Churn.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. DATA CLEANING
# =============================================================================

df.drop(columns=['customerID'], inplace=True)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])


df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "missing values remaining")
print("\nData types:\n", df.dtypes)
print("\nClass balance:\n", df['Churn'].value_counts())

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['Churn'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No Churn', 'Churn'],
    colors=['#4C72B0', '#DD8452'], startangle=90, ax=axes[0]
)
axes[0].set_title('Churn Distribution')
axes[0].set_ylabel('')

sns.countplot(x='Churn', data=df, palette='Set2', ax=axes[1])
axes[1].set_title('Churn Count')
axes[1].set_xticklabels(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Numerical Features Distribution ---
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Churn', kde=True,
                 palette='Set2', bins=30, ax=axes[i])
    axes[i].set_title(f'{col} by Churn')
plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Boxplots for Outlier Detection ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x='Churn', y=col, palette='Set2', ax=axes[i])
    axes[i].set_xticklabels(['No Churn', 'Churn'])
    axes[i].set_title(f'{col} — Outlier Check')
plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Categorical Features vs Churn ---
cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)['Churn'].mean().reset_index()
    sns.barplot(data=churn_rate, x=col, y='Churn',
                palette='Set2', ax=axes[i])
    axes[i].set_title(f'Churn Rate by {col}')
    axes[i].set_ylabel('Churn Rate')
    axes[i].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('categorical_churn_rates.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Correlation Heatmap  ---
plt.figure(figsize=(8, 6))
corr = df[num_cols + ['Churn']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

df_fe = df.copy()

# --- 4.1 New Features ---

# Average monthly spend relative to tenure (avoids div-by-zero)
df_fe['AvgMonthlySpend'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)

# Engagement score: number of add-on services subscribed
addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies']
# These cols have values: 'Yes', 'No'
for col in addon_cols:
    df_fe[col + '_bin'] = (df_fe[col] == 'Yes').astype(int)

df_fe['AddonCount'] = df_fe[[c + '_bin' for c in addon_cols]].sum(axis=1)

# Tenure bucket (bins: new, mid, long-term)
df_fe['TenureBucket'] = pd.cut(df_fe['tenure'],
                                bins=[0, 12, 36, 72],
                                labels=['New', 'Mid', 'LongTerm'])

# High spender flag (above median monthly charges)
median_charge = df_fe['MonthlyCharges'].median()
df_fe['HighSpender'] = (df_fe['MonthlyCharges'] > median_charge).astype(int)

print("\nNew features added: AvgMonthlySpend, AddonCount, TenureBucket, HighSpender")
print(df_fe[['AvgMonthlySpend', 'AddonCount', 'TenureBucket', 'HighSpender']].head())

# =============================================================================
# 5. PREPARE FEATURES & TARGET — PIPELINE
# =============================================================================

# Separate target
X = df_fe.drop(columns='Churn')
y = df_fe['Churn']

# Drop original addon cols (replaced by binaries) and tenure (bucketed)
X.drop(columns=addon_cols, inplace=True)

# Define feature groups
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                      'AvgMonthlySpend', 'AddonCount']

binary_features = ['SeniorCitizen', 'HighSpender'] + [c + '_bin' for c in addon_cols]

categorical_features = ['gender', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines', 'InternetService',
                        'Contract', 'PaperlessBilling', 'PaymentMethod',
                        'TenureBucket']

# --- Pipelines ---
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Binary features only need imputation (already 0/1)
bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features),
    ('bin', bin_pipeline, binary_features)
])

# --- Fit & Transform ---
X_preprocessed = preprocessor.fit_transform(X)

# --- Recover feature names ---
ohe_cols = preprocessor.named_transformers_['cat']['encoder']\
               .get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + ohe_cols + binary_features

preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_feature_names)
preprocessed_df['Churn'] = y.values

print("\nPreprocessed DataFrame shape:", preprocessed_df.shape)
print(preprocessed_df.head())

# =============================================================================
# 6. SAVE — Ready for SageMaker (CSV, target column first)
# =============================================================================

# SageMaker built-in algorithms expect target as the FIRST column
sagemaker_df = preprocessed_df[['Churn'] + [c for c in preprocessed_df.columns if c != 'Churn']]
sagemaker_df.to_csv('telco_churn_preprocessed.csv', index=False, header=False)

print("\n✅ Saved: telco_churn_preprocessed.csv")
print("Shape:", sagemaker_df.shape)
print("\nUpload this file to your S3 bucket using:")
print("""
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    'telco_churn_preprocessed.csv',
    'your-bucket-name',
    'data/telco_churn_preprocessed.csv'
)
print('Upload complete.')
""")

# =============================================================================
# 7. LOAD PREPROCESSED DATA
# =============================================================================
# Load the preprocessed CSV saved from the pipeline script
# (telco_churn_preprocessed.csv has NO header, target is first column)

preprocessed_df = pd.read_csv('telco_churn_preprocessed.csv', header=None)

# Separate features and target
X = preprocessed_df.iloc[:, 1:].values.astype(np.float32)
y = preprocessed_df.iloc[:, 0].values.astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution — 0:", (y == 0).sum(), "| 1:", (y == 1).sum())

# =============================================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# =============================================================================
#  CLASS WEIGHTS
# =============================================================================

# Ratio of majority to minority class
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
print('neg = ',neg)
print('pos = ',pos)
sample_weights = np.where(y_train == 1, neg / pos, 1.0)
# neg/pos = 3635/1310 ≈ 2.77
print (sample_weights)

# =============================================================================
# 9. TABNET MODEL DEFINITION
# =============================================================================

tabnet = TabNetClassifier(
    n_d=32,                  # Width of the decision step output (embedding dimension)
    n_a=32,                  # Width of the attention embedding (usually same as n_d)
    n_steps=5,               # Number of sequential attention steps
    gamma=1.3,               # Coefficient for feature reusage across steps
    n_independent=2,         # Number of independent GLU layers per step
    n_shared=2,              # Number of shared GLU layers across steps
    momentum=0.02,           # BatchNorm momentum
    epsilon=1e-15,           # Numerical stability in loss
    seed=42,
    clip_value=2,            # Gradient clipping
    lambda_sparse=1e-4,      # Sparsity regularization on feature selection
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=10, gamma=0.9),
    verbose=1,              # Print every 10 epochs
    device_name='auto'       # Uses GPU if available, else CPU
)

# =============================================================================
# 10. TRAINING
# =============================================================================

class AUCHistory(Callback):
    def __init__(self):
        self.val_auc_history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs and 'val_auc' in logs:
            self.val_auc_history.append(logs['val_auc'])

auc_callback = AUCHistory()

tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=['val'],
    eval_metric=['auc'],
    max_epochs=10,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    weights=sample_weights,
    drop_last=False,
    callbacks=[auc_callback]
)

# =============================================================================
# 11. EVALUATION
# =============================================================================

# --- Predictions ---
y_pred       = tabnet.predict(X_test)
y_pred_proba = tabnet.predict_proba(X_test)[:, 1]

# --- Metrics ---
print("\n" + "="*55)
print("CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_prec = average_precision_score(y_test, y_pred_proba)
print(f"ROC-AUC Score : {roc_auc:.4f}")
print(f"Avg Precision : {avg_prec:.4f}")


# --- 6.1 Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix — TabNet')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('tabnet_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 6.2 ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — TabNet')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('tabnet_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 6.3 Precision-Recall Curve ---
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, color='#DD8452', lw=2,
         label=f'PR Curve (AP = {avg_prec:.4f})')
plt.axhline(y=pos / len(y_test), color='k', linestyle='--',
            lw=1.5, label='Baseline (class ratio)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — TabNet')
plt.legend()
plt.tight_layout()
plt.savefig('tabnet_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 6.4 Validation Accuracy ---

val_auc_scores = auc_callback.val_auc_history

plt.figure(figsize=(8, 5))
plt.plot(val_auc_scores, color='#4C72B0', lw=2, marker='o', label='Val AUC')
plt.xticks(range(len(val_auc_scores)),
           [f'Epoch {i}' for i in range(len(val_auc_scores))])
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('TabNet Validation AUC per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('tabnet_training_history.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 6.5 Feature Importance (TabNet attention weights) ---
feature_importances = tabnet.feature_importances_

num_feat_names = numerical_features
ohe_feat_names = preprocessor.named_transformers_['cat']['encoder']\
                     .get_feature_names_out(categorical_features).tolist()
bin_feat_names = binary_features

feat_names = num_feat_names + ohe_feat_names + bin_feat_names

feat_imp_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 20 Feature Importances — TabNet Attention')
plt.tight_layout()
plt.savefig('tabnet_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 6.6 SHAP Values for Class-Specific Importance ---

background = shap.sample(X_train, 100)  # 100 samples as background

explainer = shap.KernelExplainer(tabnet.predict_proba, background)
shap_values = explainer.shap_values(X_test[:100], nsamples=100)

shap_array = np.array(shap_values)  # shape: (2, 39, 2) or similar
print("Full shap_array shape:", shap_array.shape)  # add this to confirm

shap_churn    = shap_array[1].T   # shape becomes (100, 39)
shap_no_churn = shap_array[0].T   # shape becomes (100, 39)

# --- SHAP Summary Plot for Churn=1 ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_churn, X_test[:100],
                  feature_names=feat_names,
                  plot_type='bar', show=False)
plt.title('SHAP Feature Importance — Churn=1 (Who Churns)')
plt.tight_layout()
plt.savefig('shap_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# --- SHAP Summary Plot for Churn=0 ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_no_churn, X_test[:100],
                  feature_names=feat_names,
                  plot_type='bar', show=False)
plt.title('SHAP Feature Importance — Churn=0 (Who Stays)')
plt.tight_layout()
plt.savefig('shap_no_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# --- SHAP Dot Plot for Churn=1 (shows direction + magnitude) ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_churn, X_test[:100],
                  feature_names=feat_names,
                  plot_type='dot', show=False)
plt.title('SHAP Dot Plot — Feature Impact on Churn=1')
plt.tight_layout()
plt.savefig('shap_dot_churn.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. SAVE MODEL
# =============================================================================

tabnet.save_model('tabnet_churn_model')
print("\n✅ Model saved as: tabnet_churn_model.zip")
print("To load it later: tabnet.load_model('tabnet_churn_model.zip')")