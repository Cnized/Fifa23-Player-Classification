import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.chdir(r"C:\Users\Kian K\Downloads\داده کاوی\Project")

df = pd.read_csv("data/fifa_cleaned.csv")

feature_cols = ['PaceTotal', 'ShootingTotal', 'PassingTotal', 'DribblingTotal',
                'DefendingTotal', 'PhysicalityTotal', 'Overall', 'Potential', 'Age']

df_gk = df[df['Position'] == 'Goalkeeper'].copy()
df_field = df[df['Position'] != 'Goalkeeper'].copy()

X_field = df_field[feature_cols].values

inertia = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_field)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_field, km.labels_))

best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_field['Cluster'] = km_final.fit_predict(X_field)
df_gk['Cluster'] = 3

df_all = pd.concat([df_field, df_gk], ignore_index=True)

cluster_summary = df_field.groupby('Cluster')[feature_cols].mean().round(3)
print("Field Players Cluster Summary:")
print(cluster_summary)

# Cluster 0: High Defending(0.61) + Physical(0.61), Low Shooting → Defensive Players
# Cluster 1: Highest Overall(0.56) + high in all stats → Elite All-Rounders
# Cluster 2: High Pace(0.65) + Shooting(0.57) + Dribbling(0.55), Low Defending → Attacking Players
# Cluster 3: Goalkeepers (separated manually)
cluster_labels = {
    0: 'Defensive Players',
    1: 'Elite All-Rounders',
    2: 'Attacking Players',
    3: 'Goalkeepers'
}

df_all['Cluster_Label'] = df_all['Cluster'].map(cluster_labels)

print("\nCluster Distribution:")
print(df_all['Cluster_Label'].value_counts())

print("\nPosition vs Cluster:")
print(pd.crosstab(df_all['Position'], df_all['Cluster_Label']))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_all[feature_cols].values)
df_all['PCA1'] = X_pca[:, 0]
df_all['PCA2'] = X_pca[:, 1]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Phase 4 — Clustering Results (K-Means)', fontsize=16, fontweight='bold')

axes[0, 0].plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0, 0].set_title('Elbow Method — Finding Optimal K')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()

axes[0, 1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0, 1].set_title('Silhouette Score vs K')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for cluster_id in range(4):
    mask = df_all['Cluster'] == cluster_id
    axes[0, 2].scatter(df_all.loc[mask, 'PCA1'], df_all.loc[mask, 'PCA2'],
                       c=colors[cluster_id], label=cluster_labels[cluster_id],
                       alpha=0.6, s=20)
axes[0, 2].set_title('Player Clusters (PCA Visualization)')
axes[0, 2].set_xlabel('PCA Component 1')
axes[0, 2].set_ylabel('PCA Component 2')
axes[0, 2].legend(fontsize=8)

cluster_counts = df_all['Cluster_Label'].value_counts()
axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='white')
axes[1, 0].set_title('Cluster Size Distribution')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=15)

radar_cols = ['PaceTotal', 'ShootingTotal', 'PassingTotal', 'DefendingTotal', 'PhysicalityTotal']
full_summary = df_all.groupby('Cluster')[radar_cols].mean()
x_pos = np.arange(len(radar_cols))
for i in range(4):
    axes[1, 1].plot(x_pos, full_summary.loc[i].values, 'o-', color=colors[i],
                    label=cluster_labels[i], linewidth=2, markersize=6)
axes[1, 1].set_title('Cluster Profiles (Average Stats)')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(['Pace', 'Shooting', 'Passing', 'Defending', 'Physical'], rotation=15)
axes[1, 1].set_ylabel('Normalized Value (0-1)')
axes[1, 1].legend(fontsize=8)

crosstab = pd.crosstab(df_all['Position'], df_all['Cluster_Label'])
sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 2])
axes[1, 2].set_title('Position vs Cluster Heatmap')
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Position')

plt.tight_layout()
plt.savefig('phase4_charts.png', dpi=150, bbox_inches='tight')
plt.show()

df_all.to_csv('data/fifa_clustered.csv', index=False)
print("\nSaved: phase4_charts.png")
print("Saved: data/fifa_clustered.csv")
print("Phase 4 Done!")
