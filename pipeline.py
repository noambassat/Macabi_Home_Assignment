# Auto-generated from macabi_home_assignment.ipynb
# Download Hebrew font for plotting (Alef-Regular)

# Install required packages quietly


import warnings
import re
from collections import defaultdict
from tqdm import tqdm


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from bidi.algorithm import get_display


from ydata_profiling import ProfileReport


from scipy import stats
from scipy.stats import ttest_ind, pointbiserialr

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import shap



from sentence_transformers import SentenceTransformer
import torch

warnings.filterwarnings('ignore')

df = pd.read_csv('ds_assignment_data.csv')
df.head(3).T

df.info()

df["Y"].value_counts()

df.dtypes.value_counts()

def categorize_columns(df):
    group1, group2, group3 = [], [], []

    for col in df.columns:
        if col == "Y" or col.startswith("match_"):
            group1.append(col)
        elif col.endswith("_sum"):
            group2.append(col)
        else:
            group3.append(col)

    prefix_groups = defaultdict(list)
    for col in group3:
        prefix = col.split("_")[0]
        prefix_groups[prefix].append(col)

    return group1, group2, group3, prefix_groups

group1, group2, group3, prefix_groups = categorize_columns(df)

def print_column_groups(group1, group2, prefix_groups):
    print("== Group 1: Y and match_* ==")
    for col in group1: print(f"- {col}")

    print("\n== Group 2: *_sum ==")
    for col in group2: print(f"- {col}")

    print("\n== Group 3 by prefix ==")
    for prefix, cols in prefix_groups.items():
        print(f"{prefix} ({len(cols)}): {cols}\n")

print_column_groups(group1, group2, prefix_groups)

df[group3].duplicated().sum()

def initial_analysis(df, columns):
    drop_cols = []

    print(df[columns].info())
    for col in columns:
        print(f"\n==={col}===")
        display(df[col].value_counts(dropna=False))
        if df[col].nunique(dropna=False) == 1:
            drop_cols.append(col)
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            print(df[col].describe())

    return [c for c in columns if c not in drop_cols], drop_cols

group1, drop_cols1 = initial_analysis(df, group1)

drop_cols_all = drop_cols1 + drop_cols2
df.drop(columns=drop_cols1, inplace=True)
print(f"Dropped columns: {drop_cols1}")

group1_counts = pd.DataFrame(df[group1].value_counts().sort_values(ascending=False))
group1_counts

# How many Y=1 have each match flag
df[df["Y"] == 1][group1].sum().sort_values()

match_counts = df.loc[df["Y"] == 1, group1[:-1]].sum(axis=1)
match_counts.value_counts().sort_index()

group1

# Check for Y=1 rows with no match flags
df["no_match"] = (df[group1].sum(axis=1) == 0)
df[df["Y"] == 1]["no_match"].sum()

# Check for Y=0 rows with match flags
df[(df["Y"] == 0) & (df[group1[:-1]].sum(axis=1) > 0)]


group2, drop_cols2 = initial_analysis(df, group2)

drop_cols2
df.drop(columns=drop_cols2, inplace=True)
print(f"Dropped columns: {drop_cols2}")

# Count unique combinations of the group2 columns
value_counts_df = df[group2].value_counts().reset_index(name='count')

value_counts_df.sort_values(by='count', ascending=False).sort_values(by='eclampsia_sum', ascending=False)

group_2_corr = df[["essential_hypertension_sum", "pregnancy_hypertension_sum", "preeclampsia_sum", "eclampsia_sum", "labs_sum"]].corr().abs()
sns.heatmap(group_2_corr, annot=True)

df[(df["pregnancy_hypertension_sum"]==1) & (df["essential_hypertension_sum"]==1)].shape[0] # how many womam are both ?

df[(df["Y"] == 0) & (df[group2].sum(axis=1) > 0)]

group2

indicator_cols = group1 + group2
indicator_cols.remove('Y')
indicator_cols.remove('match_diag_141')
indicator_cols

prefix_groups.keys()

def drop_constant_columns(df, columns, prefix_groups):
    # Identify constant columns
    to_remove = [col for col in columns if df[col].nunique(dropna=False) <= 1]
    print("Constant columns to remove:", to_remove)

    # Drop from DataFrame
    df.drop(columns=to_remove, inplace=True)

    # Remove from columns list
    columns = [col for col in columns if col not in to_remove]

    # Update prefix_groups accordingly
    for prefix, cols in prefix_groups.items():
        prefix_groups[prefix] = [col for col in cols if col not in to_remove]

    return columns, prefix_groups

group3, prefix_groups = drop_constant_columns(df, group3, prefix_groups)

def plot_filtered_correlation_heatmap(df, columns=group3, threshold=0.8, method='pearson'):
    """
    Plots a full heatmap of correlations for features that have at least one strong correlation
    (excluding self-correlation). Keeps only features with at least 2 correlations ≥ threshold
    including the diagonal (self-correlation).

    Parameters:
    - df: DataFrame with data
    - columns: list of columns to include (numeric only)
    - threshold: minimum absolute correlation to include (e.g. 0.8)
    - method: 'pearson', 'spearman', or 'kendall'
    """
    # Select numeric subset
    if columns is None:
        df_corr = df.select_dtypes(include=['float64', 'int64'])
    else:
        df_corr = df[columns].select_dtypes(include=['float64', 'int64'])

    # Compute correlation matrix
    corr = df_corr.corr(method=method)

    # Count number of strong correlations (|r| ≥ threshold), including self (diagonal = 1.0)
    abs_corr = corr.abs()
    strong_counts = (abs_corr >= threshold).sum(axis=1)

    # Keep only features with at least 2 strong correlations (self + at least one other)
    keep_features = strong_counts[strong_counts >= 2].index
    corr_filtered = corr.loc[keep_features, keep_features]

    if corr_filtered.empty:
        print(f"No features with ≥1 strong correlation (|r| ≥ {threshold}).")
        return

    # Plot full heatmap (no mask)
    plt.figure(figsize=(max(10, 0.5 * len(corr_filtered)), max(6, 0.5 * len(corr_filtered))))
    sns.heatmap(corr_filtered, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, fmt=".2f")
    plt.title(f'Correlation Heatmap | ≥1 strong correlation (|r| ≥ {threshold})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(df, columns, lower=0.01, upper=0.99):
    for col in columns:
        # percentile bounds
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)

        # histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col)
        plt.title(col)
        plt.tight_layout()
        plt.show()

        # boxplot with percentile lines
        plt.figure(figsize=(6, 3))
        sns.boxplot(data=df, x=col)
        plt.axvline(lower_bound, color='red', linestyle='--')
        plt.axvline(upper_bound, color='red', linestyle='--')
        plt.title(col)
        plt.tight_layout()
        plt.show()

def plot_feature_by_group(df, x_col, y_col, title=None):
    """
    Plots a boxplot using the given DataFrame, X and Y column names.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column name for the X-axis (categorical).
        y_col (str): The column name for the Y-axis (numeric).
        title (str, optional): Title of the plot. Defaults to auto-generated.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=x_col, y=y_col, data=df)

    if title is None:
        title = f'{y_col} Distribution by {x_col}'

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()


    print("\n"*3)

    plt.figure(figsize=(6,4))
    sns.barplot(x=x_col, y=y_col, data=df)
    plt.title(f'{y_col} by {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(f'{y_col}')
    plt.show()

def plot_target_distribution_by_missingness(df, missing_col, target_col='Y', figsize=(6,4), cmap='Set2'):
    dist = df.groupby(df[missing_col].isna())[target_col].value_counts(normalize=True).unstack()

    dist.plot(kind='bar', stacked=True, figsize=figsize, colormap=cmap)

    plt.title(f"Distribution of '{target_col}' by '{missing_col}' Missingness")
    plt.xlabel(f"'{missing_col}' is Missing")
    plt.ylabel("Proportion")
    plt.legend(title=target_col, bbox_to_anchor=(1, 1))
    plt.xticks(ticks=[0,1], labels=['Present', 'Missing'], rotation=0)
    plt.tight_layout()
    plt.show()

df[prefix_groups["int"]].info()

df[prefix_groups["int"]].describe()

# Attempt to interpret int_date as datetime
df['int_date_dt'] = pd.to_datetime(df['int_date'], origin='1899-12-30', unit='D')
df['int_date_dt'].describe()

# Based on the range of years (e.g. 1900 to 2147), this column does not represent a valid date.

df.drop(columns=['int_date', 'int_date_dt'], inplace=True)

prefix_groups['int'].remove('int_date')
group3.remove('int_date')

df[prefix_groups['demog']].info()

df[prefix_groups['demog']].describe()

df.groupby('Y')['demog_customer_age'].describe()

plot_feature_by_group(df, 'Y', 'demog_customer_age')

 plot_feature_distribution(df, prefix_groups['demog'], lower=0.01, upper=0.99)

df["demog_capitationcoefficient"].unique()

print("Distribution of Y by whether 'demog_customer_age' is missing:")
df.groupby(df['demog_customer_age'].isna())['Y'].value_counts(normalize=True)

plot_target_distribution_by_missingness(df, missing_col='demog_customer_age')

print("Summary statistics for 'demog_capitationcoefficient' grouped by missingness of 'demog_customer_age':")
(df.groupby(df['demog_customer_age'].isna())['demog_capitationcoefficient'].describe())

df[df['demog_customer_age'].isna() & df['demog_capitationcoefficient'].isna()].shape

df[df['demog_customer_age'].isna()].index == df[df['demog_capitationcoefficient'].isna()].index

df[df['demog_customer_age'].isna()]['Y'].value_counts()

df.dropna(subset=['demog_customer_age', 'demog_capitationcoefficient'],inplace=True)

smoking_cols = prefix_groups["smoking"]

df[smoking_cols].info()

df[smoking_cols].describe()

df['smoking_is_smoker'].value_counts(dropna=False)

df['smoking_is_smoker'] = df['smoking_is_smoker'].astype('category')

df.groupby(df['smoking_is_smoker'].isna())['Y'].value_counts(normalize=True)

df.groupby(df['smoking_is_smoker'].isna())['demog_customer_age'].describe()

df.groupby(df['smoking_is_smoker'].isna())['demog_capitationcoefficient'].describe()

plot_feature_by_group(df, 'smoking_is_smoker', 'Y')

plot_target_distribution_by_missingness(df, missing_col='smoking_is_smoker')

df.groupby('smoking_is_smoker')['smoking_smoking_years'].describe()

df['smoking_total_heavy_smokers'].value_counts().sort_index()

df.groupby('smoking_is_smoker')['smoking_total_heavy_smokers'].mean()

df.groupby(df['smoking_total_heavy_smokers'] > 0)['Y'].mean()

df[(df['smoking_is_smoker'] != 0) & (df['smoking_smoking_years'] == 0)][smoking_cols].shape

df[(df['smoking_is_smoker'].isna()) & (~df['smoking_smoking_years'].isna())][smoking_cols]

df[(df['smoking_is_smoker'] == 0) & (df['smoking_smoking_years'] != 0)][smoking_cols]

plot_feature_by_group(df, 'smoking_is_smoker', 'smoking_smoking_years')

plot_feature_by_group(df, 'Y', 'demog_customer_age')

plt.figure(figsize=(6,4))
sns.histplot(df['smoking_smoking_years'], bins=30, kde=True)
plt.title('Histogram of Smoking Years')
plt.xlabel('Smoking Years')
plt.ylabel('Count')
plt.show()


df[df['smoking_smoking_years'] > 40][['smoking_smoking_years']].shape[0]

(df['smoking_smoking_years'].isna() & (df['smoking_is_smoker'].notna())).sum()

df.loc[df['smoking_smoking_years'] > 50, 'smoking_smoking_years'] = np.nan

df.loc[(df['smoking_is_smoker'].isin([1, 2])) & (df['smoking_smoking_years'] == 0), 'smoking_smoking_years'] = np.nan

plt.figure(figsize=(6,4))
sns.histplot(df['smoking_smoking_years'].dropna(), bins=30, kde=True)
plt.title('Histogram of Smoking Years')
plt.xlabel('Smoking Years')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='smoking_is_smoker', y='smoking_smoking_years', inner='quartile')
plt.title('Smoking Years Distribution by Smoker Status')
plt.xlabel('smoking_is_smoker')
plt.ylabel('smoking_smoking_years')
plt.tight_layout()
plt.show()

(df['smoking_smoking_years'].isna() & (df['smoking_is_smoker'].notna())).sum()

mask = df['smoking_smoking_years'].isna() & df['smoking_is_smoker'].notna()
df.loc[mask, 'Y'].value_counts(normalize=True)

df[prefix_groups['lab']].info()

df[prefix_groups['lab']].describe().T

plot_feature_distribution(df, prefix_groups['lab'])

plot_filtered_correlation_heatmap(df,prefix_groups['lab'], threshold=0.8, method='pearson')

df[prefix_groups['measure']].info()

df.groupby(df['measure_blood_pressure_sys_stddev_val'].isna())['Y'].value_counts(normalize=True)

sns.boxplot(x=df['measure_blood_pressure_sys_stddev_val'].isna(), y=df['demog_customer_age'])

df[prefix_groups['measure']].describe().T

plot_feature_distribution(df, prefix_groups['measure'])

plot_filtered_correlation_heatmap(df, columns=prefix_groups['measure'], threshold=0.8, method='pearson')

group_4_24 = prefix_groups['4'] + prefix_groups['24']

df[group_4_24].info()

total_rows = df.shape[0]

non_missing = df[group_4_24].notna().sum()
coverage = (non_missing / total_rows * 100).round(2)

low_coverage = pd.DataFrame({
    'non_missing_count': non_missing,
    'coverage_percent': coverage
}).sort_values('non_missing_count')

low_coverage = low_coverage[low_coverage['non_missing_count'] < 100]

print(low_coverage)

# Calculate Y==1 stats for each column in low_coverage
y_1_percent = {}
y_1_count = {}
y_0_count = {}

for col in low_coverage.index.tolist():
    valid_rows = df[col].notna()
    if valid_rows.sum() > 0:
        y_valid = df.loc[valid_rows, 'Y']
        y_1_count[col] = int((y_valid == 1).sum())
        y_0_count[col] = int((y_valid == 0).sum())
        y_1_percent[col] = round(y_valid.mean() * 100, 2)
    else:
        y_1_count[col] = 0
        y_0_count[col] = 0
        y_1_percent[col] = None

# Add new columns to the table
low_coverage['Y==1_count'] = pd.Series(y_1_count)
low_coverage['Y==0_count'] = pd.Series(y_0_count)
low_coverage['Y==1_percent'] = pd.Series(y_1_percent)

# Display updated table
display(low_coverage)

low_coverage = low_coverage[low_coverage['non_missing_count'] < 10] # drop less than 10 not-null

cols_to_drop = low_coverage.index.tolist()
cols_to_drop

df.drop(columns=cols_to_drop, inplace=True)
group_4_24 = [col for col in group_4_24 if col not in cols_to_drop]
group3 = [col for col in group3 if col not in cols_to_drop]

prefix_groups["4"] = [col for col in prefix_groups["4"] if col not in cols_to_drop]
prefix_groups["24"] = [col for col in prefix_groups["24"] if col not in cols_to_drop]

# Create two feature groups based on column suffix
num_of_diag_cols = [col for col in group_4_24 if col.endswith('_num_of_diag')]
days_since_diag_cols = [col for col in group_4_24 if col.endswith('_days_since_last_diag')]

df[num_of_diag_cols].describe().T

df['24_diag_80_num_of_diag'].value_counts(bins=[0, 1, 5, 10, 20, 50, 100, np.inf])

df.loc[df['24_diag_80_num_of_diag'] > 50, ['Y', '24_diag_80_num_of_diag']]

plot_feature_by_group(df, 'Y', '24_diag_80_num_of_diag')

check = df.loc[df['24_diag_80_num_of_diag'] > 100]
check

for text in check['clinical_sheet']: print(text)

df.loc[df['24_diag_80_num_of_diag'] > 50, '24_diag_80_num_of_diag'] = np.nan

def plot_diag_distribution(df, group):
  for column in group:
    counts = df.loc[df[column] != 0, column].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f'Distribution of {column} without zero values')
    plt.xlabel('Number of diagnoses')
    plt.ylabel('Number of patients')
    plt.tight_layout()
    plt.show()


plot_diag_distribution(df,num_of_diag_cols)

df['total_diag_count'] = df[num_of_diag_cols].sum(axis=1)

sns.barplot(x='Y', y='total_diag_count', data=df)
plt.title("Total Diagnosis Count by Y")

df.groupby('Y')['total_diag_count'].agg(['mean', 'std', 'median', 'max'])

group_0 = df[df['Y'] == 0]['total_diag_count']
group_1 = df[df['Y'] == 1]['total_diag_count']
stat, p = ttest_ind(group_0, group_1, equal_var=False)
print(f"t-test p-value = {p:.4e}")

df[days_since_diag_cols].describe().T

plot_feature_distribution(df, days_since_diag_cols)

violations_4 = []
violations_24 = []

for col in df.columns:
    if col.startswith('4_diag_') and col.endswith('_days_since_last_diag'):
        mask = df[col] > 120
        if mask.any():
            violations_4.append((col, mask.sum()))
    elif col.startswith('24_diag_') and col.endswith('_days_since_last_diag'):
        mask = df[col] > 730
        if mask.any():
            violations_24.append((col, mask.sum()))

print("\nViolations of rule for 4_diag (>120 days):")
for v in violations_4:
    print(f"{v[0]}: {v[1]} rows")

print("\nViolations of rule for 24_diag (>730 days):")
for v in violations_24:
    print(f"{v[0]}: {v[1]} rows")

df.loc[df['4_diag_94_days_since_last_diag'] > 120, ['4_diag_94_days_since_last_diag','Y']]

df.loc[df['24_diag_61_days_since_last_diag'] > 730, ['24_diag_61_days_since_last_diag','Y']]


df_melted = df[days_since_diag_cols + ['Y']].melt(id_vars='Y')

plt.figure(figsize=(12, max(6, len(days_since_diag_cols) * 0.3)))
sns.boxplot(x='value', y='variable', hue='Y', data=df_melted, orient='h', showfliers=False)
plt.title('Distribution of Days Since Last Diagnosis (by Y)')
plt.tight_layout()
plt.show()

plot_filtered_correlation_heatmap(df, columns=days_since_diag_cols, threshold=0.8, method='pearson')

group3.remove('clinical_sheet')

col1 = 'pregnancy_hypertension_sum'
col2 = 'essential_hypertension_sum'


overlap_1_in_2 = df[df[col1] == 1][col2].value_counts(normalize=True)


overlap_2_in_1 = df[df[col2] == 1][col1].value_counts(normalize=True)


print(f"\nAmong {col1}=1:")
print(overlap_1_in_2)

print(f"\nAmong {col2}=1:")
print(overlap_2_in_1)

pd.crosstab(df[col1], df[col2], normalize='index')

# This function assigns each patient a single severity level and a single diagnosis source, based on priority.
# If multiple conditions or sources exist, only the most severe or first-matching source is retained.


def add_severity_and_source(df):
    # Severity clearly defined medically
    severity_conditions = [
        ('eclampsia_sum', 'Eclampsia'),
        ('preeclampsia_sum', 'Preeclampsia'),
        ('labs_sum', 'Lab-based diagnosis'),
        ('essential_hypertension_sum', 'Essential Hypertension'),
        ('pregnancy_hypertension_sum', 'Pregnancy Hypertension')
    ]

    df['severity_level'] = 'No Diagnosis'
    for col, level in severity_conditions:
        df.loc[(df[col] == 1) & (df['severity_level'] == 'No Diagnosis'), 'severity_level'] = level

    # Source of diagnosis clearly separated
    source_conditions = [
        ('match_pdf_after', 'Hospital Document (PDF)'),
        ('match_rasham_after', 'Medical Registry (Rasham)'),
        ('match_aspirin_after', 'Aspirin prescription'),
        ('match_diag_141', 'Diagnosis Registry')
    ]

    df['diagnosis_source'] = 'No Diagnosis'
    for col, source in source_conditions:
        df.loc[(df[col] == 1) & (df['diagnosis_source'] == 'No Diagnosis'), 'diagnosis_source'] = source

    return df

df = add_severity_and_source(df)

# 1. Validate diagnosis_source matches Y perfectly
no_source_but_positive_Y = df[(df['diagnosis_source'] == 'No Diagnosis') & (df['Y'] != 0)]
source_but_negative_Y = df[(df['diagnosis_source'] != 'No Diagnosis') & (df['Y'] != 1)]

print(f"No source but Y=1 cases: {len(no_source_but_positive_Y)} (should be 0)")
print(f"Has source but Y=0 cases: {len(source_but_negative_Y)} (should be 0)")

# 2. Validate severity_level matches match_diag_141 and Y correctly
no_severity_but_diag141 = df[(df['severity_level'] == 'No Diagnosis') & (df['match_diag_141'] != 0)]
severity_but_no_diag141 = df[(df['severity_level'] != 'No Diagnosis') & (df['match_diag_141'] != 1)]
severity_but_negative_Y = df[(df['severity_level'] != 'No Diagnosis') & (df['Y'] != 1)]

print(f"No severity but match_diag_141=1 cases: {len(no_severity_but_diag141)} (should be 0)")
print(f"Has severity but match_diag_141=0 cases: {len(severity_but_no_diag141)} (should be 0)")
print(f"Has severity but Y=0 cases: {len(severity_but_negative_Y)} (should be 0)")

def plot_distribution(df,y_col):
    plt.figure(figsize=(10,6))
    sns.countplot(
        y=y_col,
        data=df,
        order=df[y_col].value_counts().index,
        palette='viridis'
    )
    plt.title(f'Distribution of {y_col}', fontsize=16)
    plt.xlabel('Number of Patients')
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()


# plot pie chart
def plot_pie_chart(df, y_col, no_zeros = False):
  plt.figure(figsize=(8, 8))
  counts = counts = df[y_col].value_counts()
  plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
  title = f'{y_col} Distribution'
  if no_zeros:
    title += ' (non-zero cases)'
  plt.title(title)
  plt.tight_layout()
  plt.show()

plot_pie_chart(df, 'Y')

plot_distribution(df,'severity_level')

# filter out rows with 'No Diagnosis'
severity_filtered = df[df['severity_level'] != 'No Diagnosis']


plot_distribution(severity_filtered, "severity_level")

plot_pie_chart(severity_filtered, y_col='severity_level', no_zeros=True)

df['diagnosis_source'].unique()


# filter out rows with 'No Diagnosis'
diag_filtered = df[df['diagnosis_source'] != 'No Diagnosis']

plot_distribution(diag_filtered, y_col="diagnosis_source")

plot_pie_chart(severity_filtered, y_col='severity_level', no_zeros=True)

def plot_boxplot_distribution(target_col):
  for col in ['demog_customer_age', 'demog_capitationcoefficient']:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=target_col, y=col, data=df, palette='coolwarm')
    plt.title(f'{col} Distribution by {target_col}', fontsize=16)
    plt.xticks(rotation=45)
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

plot_boxplot_distribution('Y')

plot_boxplot_distribution('severity_level')

plot_boxplot_distribution('diagnosis_source')

def smoking_status_by_target(target_col):
  smoking_pivot = pd.crosstab(df[target_col], df['smoking_is_smoker'], normalize='index')
  # smoking_pivot = smoking_pivot.reindex(severity_order)

  smoking_pivot.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Set2')
  plt.title(f'Smoking Status by {target_col}', fontsize=16)
  plt.xlabel(target_col)
  plt.ylabel('Percentage')
  plt.legend(title='Smoker Status', bbox_to_anchor=(1,1))
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

smoking_status_by_target('Y')

smoking_status_by_target('severity_level')

# smoking_status_by_target('diagnosis_source')

def Blood_Pressure_By_target(target_col):
  bp_means = df.groupby(target_col)[[
      'measure_blood_pressure_sys_mean_val',
      'measure_blood_pressure_dias_mean_val'
  ]].mean()

  plt.figure(figsize=(11,5))
  sns.heatmap(bp_means, annot=True, fmt=".1f", cmap='YlGnBu')

  plt.title(f'Mean Blood Pressure by {target_col}', fontsize=12)
  plt.xlabel('Blood Pressure Measure', fontsize=10)
  plt.ylabel(target_col, fontsize=10)

  plt.xticks(rotation=30, ha='right')

  plt.tight_layout()
  plt.show()

Blood_Pressure_By_target('Y')

Blood_Pressure_By_target('severity_level')

Blood_Pressure_By_target('diagnosis_source')

def labs_by_target(target_col, target_binary=0):
  lab_cols = [col for col in df.columns if col.startswith('lab_')]
  col_binary = (df[target_col] != target_binary).astype(int)

  correlations = {col: pointbiserialr(df[col].fillna(0), col_binary)[0] for col in lab_cols}
  top_labs = sorted(correlations, key=lambda x: abs(correlations[x]), reverse=True)[:5]

  for col in top_labs:
      plt.figure(figsize=(10, 6))
      sns.boxplot(x=target_col, y=col, data=df, palette='Spectral')
      plt.title(f'{col} by {target_col}', fontsize=16)
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()


labs_by_target('Y')


labs_by_target('severity_level', target_binary='No Diagnosis')

df['diagnosis_source'].unique()

labs_by_target('diagnosis_source', "No Diagnosis")

lab_cols = [col for col in df.columns if col.startswith('lab_')]
severity_binary = (df['severity_level'] != 'No Diagnosis').astype(int)

correlations = {col: pointbiserialr(df[col].fillna(0), severity_binary)[0] for col in lab_cols}
top_labs = sorted(correlations, reverse=True)[:5]

for col in top_labs:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='severity_level', y=col, data=df, palette='Spectral')
    plt.title(f'{col} by Severity Level', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 1. Gestational Hypertension predictors vs combined hypertension targets
gestational_hypertension_predictors = [
    'measure_blood_pressure_sys_max_val',
    'measure_blood_pressure_dias_max_val',
    'lab_weight_at_lab_time_last_value',
    'demog_customer_age',
]

# Combine hypertension columns as binary target
df['hypertension_target'] = (
    df['essential_hypertension_sum'] | df['pregnancy_hypertension_sum']
).astype(int)

# Correlation plot for gestational hypertension predictors
corr_gestational = df[gestational_hypertension_predictors].apply(
    lambda x: x.corr(df['hypertension_target'])
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=corr_gestational.values, y=corr_gestational.index, palette='viridis')
plt.title('Correlation of Predictors with Gestational Hypertension (binary)')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.grid(True, alpha=0.5)
plt.show()
print("\n"*3)


# 2. Preeclampsia predictors vs preeclampsia_sum
preeclampsia_predictors = [
    'lab_weight_at_lab_time_last_value',
    'demog_customer_age',
    'lab_papp_a_MoM_last_value',
    'lab_Protein-U_last_value',
    'lab_Mean Platelet Volume (MPV)_last_value',
    'lab_Platelets (PLT)_last_value',
    'measure_blood_pressure_sys_max_val',
    'measure_blood_pressure_dias_max_val',
]

corr_preeclampsia = df[preeclampsia_predictors].apply(
    lambda x: x.corr(df['preeclampsia_sum'])
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=corr_preeclampsia.values, y=corr_preeclampsia.index, palette='mako')
plt.title('Correlation of Predictors with Preeclampsia')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.grid(True, alpha=0.5)
plt.show()
print("\n"*3)



# 3. Eclampsia predictors vs eclampsia_sum
eclampsia_predictors = [
    'demog_customer_age',
    'lab_weight_at_lab_time_last_value',
    'lab_Hemoglobin (HGB)_last_value',
    'lab_Hematocrit (HCT)_last_value',
    'measure_blood_pressure_sys_max_val',
    'measure_blood_pressure_dias_max_val',
]

corr_eclampsia = df[eclampsia_predictors].apply(
    lambda x: x.corr(df['eclampsia_sum'])
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=corr_eclampsia.values, y=corr_eclampsia.index, palette='rocket')
plt.title('Correlation of Predictors with Eclampsia')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.grid(True, alpha=0.5)
plt.show()
print("\n"*3)


# 4. All predictors vs general outcome Y (excluding target columns themselves)
all_relevant_predictors = list(set(
    gestational_hypertension_predictors +
    preeclampsia_predictors +
    eclampsia_predictors
))

# Remove target columns from all predictors if they exist
targets = [
    'essential_hypertension_sum',
    'pregnancy_hypertension_sum',
    'preeclampsia_sum',
    'eclampsia_sum'
]

final_predictors = [col for col in all_relevant_predictors if col not in targets]

corr_Y = df[final_predictors].apply(lambda x: x.corr(df['Y'])).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=corr_Y.values, y=corr_Y.index, palette='coolwarm')
plt.title('Correlation of All Relevant Predictors with Outcome (Y)')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.grid(True, alpha=0.5)
plt.show()

df["clinical_sheet"].isna().sum()

df['clinical_length'] = df['clinical_sheet'].str.len()
df.groupby(df[indicator_cols].sum(axis=1) > 0)['clinical_length'].mean()

def print_cleaned_examples(df, label_value,clinical_col="clinical_sheet", n=3):
    examples = df[df['Y'] == label_value][clinical_col].dropna().sample(n, random_state=42)
    print(f"\n=== {n} Examples where Y = {label_value} ===\n")
    for i, text in enumerate(examples, start=1):
        print(f"--- Example {i} ---\n{text.strip()}\n")

print_cleaned_examples(df, label_value=1, n=3)
print_cleaned_examples(df, label_value=0, n=3)

# Compute TF-IDF
vectorizer = TfidfVectorizer(use_idf=True, min_df=5, sublinear_tf=True)
vec_sparse = vectorizer.fit_transform(df['clinical_sheet'])
vec_df = pd.DataFrame(vec_sparse.todense(), columns=vectorizer.get_feature_names_out())

vec_df.T

chi2(vec_df[['כספית']], df['Y']==1)

# Right value is the p-value:
#  a very low p-value indicates that the word "כספית" ("Mercury") is significantly correlated with cases labeled as "sick" (Y=1),
#  distinguishing them clearly from non-sick cases.



chi2(vec_df[["דם"]],df["Y"]==1)



# Reset the index of df
df_reset = df.reset_index(drop=True)

# Chi2 scores and p-values
chi2_scores, p_values = chi2(vec_df, df_reset['Y'])

# Create DataFrame with results
chi2_df = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'chi2_score': chi2_scores,
    'p_value': p_values,
})

chi2_df


# Add average TF-IDF per class
chi2_df['tfidf_mean_Y1'] = vec_df[df_reset['Y'] == 1].mean(axis=0).values
chi2_df['tfidf_mean_Y0'] = vec_df[df_reset['Y'] == 0].mean(axis=0).values

# Clearly assign each word to the class with higher mean TF-IDF
chi2_df['class'] = np.where(chi2_df['tfidf_mean_Y1'] > chi2_df['tfidf_mean_Y0'], 1, 0)


# Sort words by chi2_score for each class separately
top_words_Y1 = chi2_df[chi2_df['class'] == 1].sort_values(by='chi2_score', ascending=False).head(20)
top_words_Y0 = chi2_df[chi2_df['class'] == 0].sort_values(by='chi2_score', ascending=False).head(20)

# Display neatly
print("Top words strongly associated with Y=1:")
display(top_words_Y1[['word', 'chi2_score', 'tfidf_mean_Y1', 'tfidf_mean_Y0']])

print("\nTop words strongly associated with Y=0:")
display(top_words_Y0[['word', 'chi2_score', 'tfidf_mean_Y1', 'tfidf_mean_Y0']])


top_words_Y1 = chi2_df[chi2_df['class'] == 1].sort_values('chi2_score', ascending=False)['word'].reset_index(drop=True)
top_words_Y0 = chi2_df[chi2_df['class'] == 0].sort_values('chi2_score', ascending=False)['word'].reset_index(drop=True)


top_words_df = pd.DataFrame({
    'Y=0': top_words_Y0,
    'Y=1': top_words_Y1
})

display(top_words_df.head(20))

def plot_top_words_chi2(vec_df, Y, class_label, font_path='Alef-Regular.ttf', top_n=20):
    """
    Plot top distinguishing words for a given binary class using Chi-Squared statistics.

    Parameters:
    vec_df (DataFrame): TF-IDF vectorized words as columns.
    Y (Series): Binary target variable indicating class labels.
    class_label (int): The class label to distinguish (0 or 1).
    font_path (str): Path to font supporting Hebrew characters.
    top_n (int): Number of top words to display.
    """

    # Reset index of Y to match vec_df
    Y = Y.reset_index(drop=True)

    # Compute average TF-IDF per class
    tfidf_Y1_mean = vec_df[Y == 1].mean(axis=0)
    tfidf_Y0_mean = vec_df[Y == 0].mean(axis=0)

    # Ensure words are correctly associated to the selected class
    if class_label == 1:
        selected_words = tfidf_Y1_mean[tfidf_Y1_mean > tfidf_Y0_mean].index
    else:
        selected_words = tfidf_Y0_mean[tfidf_Y0_mean > tfidf_Y1_mean].index

    # Filter vec_df for selected words
    vec_df_selected = vec_df[selected_words]

    # Compute Chi2 for selected words
    chi_scores, p_values = chi2(vec_df_selected, Y == class_label)

    # Create DataFrame for visualization
    chi2_df = pd.DataFrame({
        'Word': selected_words,
        'Chi2': chi_scores,
        'P-value': p_values
    }).sort_values('Chi2', ascending=False).head(top_n)

    # Reverse Hebrew text ONLY for barplot display
    chi2_df['Word_display'] = chi2_df['Word'].apply(get_display)

    # Plot bar chart with reversed Hebrew text
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Chi2', y='Word_display', data=chi2_df, palette='coolwarm')
    plt.title(f"Top {top_n} Words Associated with Class (Y={class_label}) by Chi-Squared", fontsize=16)
    plt.xlabel("Chi-Squared Score")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()
    print("\n*2")

    # Prepare word frequencies WITHOUT reversing for Word Cloud
    word_freq = dict(zip(chi2_df['Word'], chi2_df['Chi2']))

    # Generate Word Cloud
    wc = WordCloud(
        width=800,
        height=600,
        background_color='white',
        colormap='viridis',
        font_path=font_path
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.title(f"Word Cloud for Class (Y={class_label}) - Chi-Squared", fontsize=16)
    plt.show()

plot_top_words_chi2(vec_df, df["Y"], class_label=1)

plot_top_words_chi2(vec_df, df["Y"], class_label=0)

selected_words = chi2_df[chi2_df['chi2_score'] > 25]['word'].tolist()

selected_words


for word in selected_words:
    df[f'word_{word}'] = df['clinical_sheet'].str.contains(word).astype(int)



# Select only the word columns
word_cols = [col for col in df.columns if col.startswith('word_')]

# Calculate point-biserial correlation clearly and professionally
correlations = []
for col in word_cols:
    corr, p_value = pointbiserialr(df[col], df['Y'])
    correlations.append({'word': col, 'correlation_with_Y': corr, 'p_value': p_value})

# Create a clear summary DataFrame and sort it by correlation
correlation_df = pd.DataFrame(correlations).sort_values(by='correlation_with_Y', ascending=False)

# Display the sorted DataFrame
display(correlation_df.head(20))

# Top 20 correlated words
top_corr_df = correlation_df

# Reverse Hebrew text if necessary (assuming Hebrew words)
top_corr_df['word_display'] = top_corr_df['word'].apply(get_display)

# Plot horizontal bar plot clearly showing correlations
plt.figure(figsize=(14, 10))
sns.barplot(
    x='correlation_with_Y',
    y='word_display',
    data=top_corr_df,
    palette='coolwarm',
    orient='h'
)
plt.xlabel('Correlation with Y', fontsize=14)
plt.ylabel('Word', fontsize=14)
plt.title('Top 20 Words Correlated with Y', fontsize=16)
plt.axvline(0, color='gray', linestyle='--')  # Clearly mark zero correlation
plt.tight_layout()
plt.show()

def extract_last_clinical_note(df, clinical_col="clinical_sheet"):

  df[clinical_col].split()



def extract_last_week_paragraph(text):
    """
    Extract the last paragraph explicitly starting with 'שבוע'.
    Returns entire text as fallback if not found.

    Parameters:
    text (str): Clinical description text.

    Returns:
    str: Last relevant paragraph or whole text if none found.
    """
    splits = re.split(r'(?=^\s*שבוע\s*\d+)', text, flags=re.MULTILINE)
    valid_splits = [split.strip() for split in splits if split.strip()]
    return valid_splits[-1] if valid_splits else text.strip()

df['last_week_paragraph'] = df['clinical_sheet'].apply(extract_last_week_paragraph)

def print_cleaned_examples(df, label_value,clinical_col="clinical_sheet", n=3):
    examples = df[df['Y'] == label_value][clinical_col].dropna().sample(n, random_state=42)
    print(f"\n=== {n} Examples where Y = {label_value} ===\n")
    for i, text in enumerate(examples, start=1):
        print(f"--- Example {i} ---\n{text.strip()}\n")

print_cleaned_examples(df, label_value=1,clinical_col="last_week_paragraph", n=3)
print_cleaned_examples(df, label_value=0,clinical_col="last_week_paragraph", n=3)

model = SentenceTransformer('intfloat/multilingual-e5-base')

model.to('cuda')

texts = df['last_week_paragraph'].tolist()

batch_size = 64

embeddings = []

# Process texts in batches
for i in tqdm(range(0, len(texts), batch_size), desc='Encoding embeddings (batch)'):
    batch_texts = texts[i:i+batch_size]

    with torch.no_grad():
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)

    embeddings.extend(batch_embeddings)

# Save embeddings back to DataFrame
df['e5_embedding'] = embeddings

original_df = df.copy()

# for col in df.columns: print(f'"{col}",')

df.drop(columns=['clinical_sheet',
                 'last_week_paragraph',
                 "match_diag_141",
                  "match_rasham_after",
                  "match_aspirin_after",
                  "match_pdf_after",
                  "essential_hypertension_sum",
                  "pregnancy_hypertension_sum",
                  "preeclampsia_sum",
                  "eclampsia_sum",
                  "labs_sum",
                  "clinical_sheet",
                  "no_match",
                  "total_diag_count",
                  "severity_level",
                  "diagnosis_source",
                  "hypertension_target"],
         inplace=True)

for col in df.columns: print(f'"{col}",')

embeddings_array = np.vstack(df['e5_embedding'].values)

numeric_data = df.drop(columns=['e5_embedding','Y'])

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

y = df['Y']

# Impute missing values clearly as -1 (a medically meaningless value)
model = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=-1),
    StandardScaler(),
    ElasticNetCV(cv=5, random_state=42)
)

# Fit model to identify relevant features
model.fit(numeric_data, y)

coefs = pd.Series(
    model.named_steps['elasticnetcv'].coef_,
    index=numeric_data.columns
)

# Identify clearly relevant/irrelevant features
selected_features = coefs[coefs.abs() > 0].index.tolist()
removed_features = coefs[coefs.abs() == 0].index.tolist()

print("Selected features by ElasticNet:", selected_features)
print("Removed features by ElasticNet:", removed_features)

selected_features = set(selected_features + final_predictors)

numeric_data_selected = numeric_data[list(selected_features)]

for col in numeric_data_selected.columns: print(col)

X = np.hstack([numeric_data, embeddings_array])

warnings.filterwarnings('ignore')

# Indicator columns used for stratification
indicator_cols = [
    'preeclampsia_sum', 'eclampsia_sum',
    'match_rasham_after', 'match_aspirin_after', 'match_pdf_after',
    'essential_hypertension_sum', 'pregnancy_hypertension_sum',
    'labs_sum'
]

# Combined stratification group
stratify_groups = original_df[indicator_cols].astype(str).agg('-'.join, axis=1)

# Class weights
class_weight_dict = {0: 1, 1: 3}

# Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Results storage
results_full = original_df.copy().reset_index(drop=True)
results_full['predicted_proba'] = np.nan

for train_idx, test_idx in skf.split(X, stratify_groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    model = lgb.LGBMClassifier(class_weight=class_weight_dict, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results_full.loc[test_idx, 'predicted_proba'] = y_pred_proba

results_full[["predicted_proba","Y"]].sort_values(by="predicted_proba",ascending=False).head(10)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y, results_full['predicted_proba'])

# Find intersection point of precision and recall
intersection = np.argmin(np.abs(precision[:-1] - recall[:-1]))
optimal_threshold = thresholds[intersection]


plt.figure(figsize=(10, 7))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue', linewidth=2)
plt.plot(thresholds, recall[:-1], label='Recall', color='red', linewidth=2)
plt.axvline(x=optimal_threshold, linestyle='--', color='green', label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.xlabel('Threshold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Precision-Recall vs Threshold', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# Show confusion matrix and metrics at selected threshold
def evaluate_threshold(threshold, y_true, y_proba):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"\nConfusion Matrix (Threshold = {threshold:.2f}):")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix at Threshold {threshold:.2f}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(report)

# Evaluate at optimal threshold
evaluate_threshold(optimal_threshold, y, results_full['predicted_proba'])

evaluate_threshold(0.05, y, results_full['predicted_proba'])

evaluate_threshold(0.08, y, results_full['predicted_proba'])

# FP and FN analysis by indicators
fp_fn_df = results_full.copy()
fp_fn_df['FP'] = ((fp_fn_df['Y'] == 0) & (fp_fn_df['predicted_proba'] >= optimal_threshold))
fp_fn_df['FN'] = ((fp_fn_df['Y'] == 1) & (fp_fn_df['predicted_proba'] < optimal_threshold))

fp_data = fp_fn_df[fp_fn_df['FP']]
fn_data = fp_fn_df[fp_fn_df['FN']]

print("Total FP:", fp_fn_df['FP'].sum())
print("Total FN:", fp_fn_df['FN'].sum())



# Calculate percentage of indicators for FN
fn_indicator_means = fn_data[indicator_cols].mean().sort_values(ascending=False) * 100


# Plot for False Negatives
plt.figure(figsize=(10, 5))
fn_indicator_means.plot(kind='bar', color='lightblue')
plt.title('Indicator Distribution for False Negatives')
plt.ylabel('Percentage (%)')
plt.xlabel('Indicators')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


fp_analysis_cols = ['demog_customer_age', 'measure_blood_pressure_sys_max_val', 'measure_blood_pressure_dias_max_val']

fp_data = fp_fn_df[fp_fn_df['FP']]

# Plot FP distributions for relevant numeric columns
for col in fp_analysis_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(fp_data[col], kde=True, color='salmon', bins=20)
    plt.title(f'Distribution of {col} in False Positives')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("\n")

    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, color='blue', bins=20)
    plt.title(f'Distribution of {col} in all data')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n"*3)

def plot_bar_feature_importance(values, title, xlabel='Importance', ylabel='Feature', palette='Blues_d'):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=values.values, y=values.index.to_series().apply(get_display), palette=palette)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("\n"*3)

def get_lgbm_feature_importance(model, feature_names, num_features=20):
    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError("Mismatch in feature and importance lengths.")
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(num_features)
    return imp_series

def get_shap_importance_per_class(model, X, y, feature_names, num_features=20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # Binary classification
        shap_values = shap_values[1]
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['y'] = y.values if isinstance(y, pd.Series) else y

    return (
        shap_df[shap_df['y'] == 0].drop(columns='y').abs().mean().sort_values(ascending=False).head(num_features),
        shap_df[shap_df['y'] == 1].drop(columns='y').abs().mean().sort_values(ascending=False).head(num_features)
    )

# Create the correct feature names clearly
numeric_feature_names = numeric_data.columns.tolist()
embedding_feature_names = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
all_feature_names = numeric_feature_names + embedding_feature_names

# Get and plot LightGBM importance
imp_lgb = get_lgbm_feature_importance(model, all_feature_names, 20)
plot_bar_feature_importance(imp_lgb, title="Top LightGBM Feature Importances", palette='coolwarm')

# Get and plot SHAP importance per class
shap_0, shap_1 = get_shap_importance_per_class(model, X, y, all_feature_names, 20)
plot_bar_feature_importance(shap_0, title="Top SHAP Features – Class 0", palette='Blues_d')
plot_bar_feature_importance(shap_1, title="Top SHAP Features – Class 1", palette='Reds_d')


# def plot_existing_model_feature_importance(model, feature_names, num_features=20):
#     """
#     Plot clearly the feature importance from an already trained LightGBM model.

#     Parameters:
#         model: Trained LightGBM model.
#         feature_names (list): List of feature names used in the model.
#         num_features (int): Number of top features to display.

#     Returns:
#         DataFrame: Top N feature importances.
#     """

#     # Extract feature importances from the existing trained model
#     importances = model.feature_importances_

#     # Ensure feature_names matches importances length
#     if len(feature_names) != len(importances):
#         raise ValueError("Length of feature_names does not match the number of features used by the model.")

#     # Create DataFrame of importances clearly
#     feature_importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': importances
#     }).sort_values(by='importance', ascending=False).head(num_features)

#     feature_importance_df['feature'] = feature_importance_df['feature'].apply(get_display)


#     # Plotting the feature importances clearly
#     plt.figure(figsize=(10, 8))
#     sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='coolwarm')
#     plt.title('Top Features Importance from Existing LightGBM Model', fontsize=16)
#     plt.xlabel('Feature Importance')
#     plt.ylabel('Feature')
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     plt.show()

#     return feature_importance_df

# # Create the correct feature names clearly
# numeric_feature_names = numeric_data.columns.tolist()
# embedding_feature_names = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
# all_feature_names = numeric_feature_names + embedding_feature_names

# # Now plot the feature importances clearly using the existing model
# top_features = plot_existing_model_feature_importance(model, all_feature_names, num_features=20)
# # print(top_features)


# def plot_shap_feature_importance_per_class_correct(model, X, y, feature_names, num_features=20):
#     """
#     Calculate and plot SHAP feature importance per class separately.

#     Parameters:
#         model: Trained LightGBM model.
#         X (DataFrame or ndarray): Feature data used for predictions.
#         y (Series or ndarray): Binary target variable.
#         feature_names (list): List of feature names.
#         num_features (int): Number of top features to display.

#     Returns:
#         tuple: Top feature importance for class 0 and class 1 as Series.
#     """

#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)

#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]  # Binary classification: class 1

#     shap_values = np.array(shap_values)

#     assert shap_values.shape == X.shape, f"Expected shap_values shape {X.shape}, got {shap_values.shape}"

#     shap_df = pd.DataFrame(shap_values, columns=feature_names)
#     shap_df['y'] = y.values if isinstance(y, pd.Series) else y

#     mean_shap_class_0 = shap_df.loc[shap_df['y'] == 0].drop(columns=['y']).abs().mean().sort_values(ascending=False)
#     mean_shap_class_1 = shap_df.loc[shap_df['y'] == 1].drop(columns=['y']).abs().mean().sort_values(ascending=False)

#     top_features_0 = mean_shap_class_0.head(num_features)
#     top_features_1 = mean_shap_class_1.head(num_features)

#     # Apply get_display for better visualization
#     top_features_0.index = top_features_0.index.to_series().apply(get_display)
#     top_features_1.index = top_features_1.index.to_series().apply(get_display)

#     # Plot for Class 0
#     plt.figure(figsize=(10, 8))
#     sns.barplot(x=top_features_0.values, y=top_features_0.index, palette='Blues_d')
#     plt.title('Top SHAP Features Importance for Class 0', fontsize=16)
#     plt.xlabel('Mean Absolute SHAP Value')
#     plt.ylabel('Feature')
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     plt.show()

#     print("\n"*3)
#     # Plot for Class 1
#     plt.figure(figsize=(10, 8))
#     sns.barplot(x=top_features_1.values, y=top_features_1.index, palette='Reds_d')
#     plt.title('Top SHAP Features Importance for Class 1', fontsize=16)
#     plt.xlabel('Mean Absolute SHAP Value')
#     plt.ylabel('Feature')
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     plt.show()

#     return top_features_0, top_features_1

# feature_importance_0, feature_importance_1 = plot_shap_feature_importance_per_class_correct(
#     model=model,
#     X=X,
#     y=y,
#     feature_names=all_feature_names,
#     num_features=20
# )


# print(feature_importance_0)
# print(feature_importance_1)

results_full.sort_values(by='predicted_proba', ascending=False).head(2).T

# results_full.head(3).T
