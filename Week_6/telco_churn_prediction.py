import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
#import torch
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

sns.set_style('whitegrid')
sns.set_theme('notebook')

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

# --- Drop irrelevant column ---
df.drop(columns=['customerID'], inplace=True)

# --- Fix TotalCharges: it's stored as object due to spaces ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- Impute TotalCharges nulls with median (only 11 rows) ---
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# --- Encode target variable ---
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# --- Fix SeniorCitizen: already 0/1 but label it clearly ---
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "missing values remaining")
print("\nData types:\n", df.dtypes)
print("\nClass balance:\n", df['Churn'].value_counts())


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# --- 3.1 Churn Distribution ---
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

# --- 3.2 Numerical Features Distribution ---
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Churn', kde=True,
                 palette='Set2', bins=30, ax=axes[i])
    axes[i].set_title(f'{col} by Churn')
plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 3.3 Boxplots for Outlier Detection ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x='Churn', y=col, palette='Set2', ax=axes[i])
    axes[i].set_xticklabels(['No Churn', 'Churn'])
    axes[i].set_title(f'{col} — Outlier Check')
plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 3.4 Categorical Features vs Churn ---
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

# --- 3.5 Correlation Heatmap (numerical) ---
plt.figure(figsize=(8, 6))
corr = df[num_cols + ['Churn']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 3.6 Tenure vs Monthly Charges (scatter) ---
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges',
                hue='Churn', palette='Set2', alpha=0.6)
plt.title('Tenure vs Monthly Charges')
plt.tight_layout()
plt.savefig('tenure_vs_charges.png', dpi=150, bbox_inches='tight')
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
# These cols have values: 'Yes', 'No', 'No internet service'
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