import sys
import seaborn as sns
import matplotlib.pyplot as plt

args = sys.argv
if len(args) != 2:
    print("Usage: python data_analysis_main.py <dataset_path>")
    sys.exit(1)
else:
    dataset_path = args[1]
    print(f"Using dataset: {dataset_path}")

from io_operations import read_from_csv

df = read_from_csv(dataset_path)

categorical_features = df.select_dtypes(exclude=["number"]).columns  
num_features = len(categorical_features)

cols = 3 
rows = -(-num_features // cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.countplot(x=df[feature], ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis="x", rotation=45)

for i in range(num_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('output/fig_0.png')

#

df = read_from_csv(dataset_path)

numerical_features = df.select_dtypes(include=["number"]).columns
num_features = len(numerical_features)

cols = 3
rows = -(-num_features // cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.histplot(df[feature], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")

for i in range(num_features, len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.savefig('output/fig_1.png')

#

df = read_from_csv(dataset_path)

plt.figure(figsize=(10, 10))
sns.barplot(
    x=df["NObeyesdad"],
    y=df["Weight"],
    errorbar="sd",
)

plt.title("Average Weight by NObeyesdad")
plt.xlabel("NObeyesdad")
plt.ylabel("Average Weight")
plt.xticks(rotation=45)
plt.savefig('output/fig_2.png')

#

df = read_from_csv("data/dataset.csv")

plt.figure(figsize=(10, 10))
sns.histplot(df, x="Weight", hue="family_history_with_overweight", kde=True, bins=30)
plt.title("Weight Distribution according to Family History with Overweight")
plt.savefig('output/fig_3.png')

#

df = read_from_csv("data/dataset.csv")

sns.catplot(
    data=df, x="NObeyesdad", y="Weight", hue="Gender",
    kind="violin", split=True,
    height=10,
    aspect=1.5
)
plt.xticks(rotation=45, ha="right")
plt.title("Weight Distribution Across Weight Categories by Gender")
plt.tight_layout(pad=2.0)
plt.savefig('output/fig_4.png')

#

import numpy as np
import pandas as pd
from scipy import stats

df = read_from_csv("data/dataset.csv")

def cramers_v(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

categorical_features = df.select_dtypes(include=['object']).columns
cramers_matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)
for i in categorical_features:
    for j in categorical_features:
        if i == j:
            cramers_matrix.loc[i,j] = 1
        else:
            table = pd.crosstab(df[i], df[j])
            cramers_matrix.loc[i,j] = cramers_v(table)

cramers_matrix = cramers_matrix.astype(float)

plt.figure(figsize=(12, 12))
sns.heatmap(cramers_matrix, cmap="coolwarm")
plt.title("Cramér's V Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/fig_5.png')

#

df = read_from_csv("data/dataset.csv")

numeric_features = df.select_dtypes(include=["number"])
correlation_matrix = df[numeric_features.columns].corr(method="pearson")

plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Pearson Correlation Matrix - Raw Data")
plt.savefig('output/fig_6_0.png')

Q1 = numeric_features.quantile(0.25)
Q3 = numeric_features.quantile(0.75)
IQR = Q3 - Q1

df_no_outliers = numeric_features[~((numeric_features < (Q1 - 1.5 * IQR)) | (numeric_features > (Q3 + 1.5 * IQR))).any(axis=1)]

correlation_matrix = df_no_outliers[numeric_features.columns].corr(method="pearson")

plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Pearson Correlation Matrix - Outliers Removed")
plt.savefig('output/fig_6_1.png')

#

df = read_from_csv("data/dataset.csv")

plt.figure(figsize=(10, 10))
sns.lmplot(
    data=df,
    x="Height",
    y="Weight",
    hue="NObeyesdad",
    palette="Set1",
    height=6,
    aspect=2,
    legend=False,
)
plt.title("Height vs Weight by NObeyesdad")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(pad=2.0)
plt.savefig('output/fig_7.png')

#

df = read_from_csv("data/dataset.csv")

plt.figure(figsize=(10, 10))
sns.boxplot(x=df["NObeyesdad"], y=df["Height"])
plt.title("Height Distribution Across Obesity Levels")
plt.xticks(rotation=45)
plt.tight_layout(pad=2.0)
plt.savefig('output/fig_8.png')

#

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

df = read_from_csv("data/dataset.csv")
df_copy = df.copy()

le = LabelEncoder()
for col in df_copy.select_dtypes(include=['object']).columns:
    df_copy[col] = le.fit_transform(df_copy[col])

df_copy = pd.get_dummies(df_copy, columns=["MTRANS", "CAEC", "CALC"], drop_first=True)

X_scaled = StandardScaler().fit_transform(df_copy.drop(columns=["NObeyesdad"]))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["NObeyesdad"] = df["NObeyesdad"]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="NObeyesdad", palette="viridis")
plt.title("PCA of Obesity Levels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Obesity Level")
plt.savefig('output/fig_9.png')

#

from sklearn.manifold import TSNE

df = read_from_csv(dataset_path)
df_copy = df.copy()

le = LabelEncoder()
for col in df_copy.select_dtypes(include=['object']).columns:
    df_copy[col] = le.fit_transform(df_copy[col])

df_copy = pd.get_dummies(df_copy, columns=["MTRANS", "CAEC", "CALC"], drop_first=True)

features = df_copy.drop(columns=["NObeyesdad"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

y = df["NObeyesdad"]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame(X_embedded, columns=["TSNE1", "TSNE2"])
tsne_df["NObeyesdad"] = y

plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="NObeyesdad", palette="viridis")
plt.title("t-SNE Visualization of Obesity Levels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Obesity Level")
plt.savefig('output/fig_10.png')

#