import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
os.chdir(r"C:\Users\Kian K\Downloads\داده کاوی\Project")

df_fifa = pd.read_csv("data/players_fifa23.csv", low_memory=False)

leagues = ['laliga', 'premier_league', 'ligue1', 'bundesliga', 'serie_a']
base_path = "data/transfermarkt-data-master"
all_transfers = []
for league in leagues:
    league_path = os.path.join(base_path, league)
    if not os.path.exists(league_path):
        print(f"Skipped: {league}")
        continue
    for file in os.listdir(league_path):
        if file.endswith(".csv"):
            try:
                df_temp = pd.read_csv(os.path.join(league_path, file), low_memory=False)
                df_temp['league'] = league
                all_transfers.append(df_temp)
            except:
                pass

df_transfer = pd.concat(all_transfers, ignore_index=True)

print("FIFA 23 Dataset:")
print(f"  Records: {df_fifa.shape[0]}")
print(f"  Columns: {df_fifa.shape[1]}")
print(f"  Column names: {list(df_fifa.columns)}")

print("\nTransfermarkt Dataset:")
print(f"  Records: {df_transfer.shape[0]}")
print(f"  Columns: {df_transfer.shape[1]}")
print(f"  Column names: {list(df_transfer.columns)}")

print("\nSample FIFA 23:")
print(df_fifa.head(3))

print("\nSample Transfermarkt:")
print(df_transfer.head(3))

numeric_cols = ['PaceTotal', 'ShootingTotal', 'PassingTotal', 'DribblingTotal',
                'DefendingTotal', 'PhysicalityTotal', 'Overall', 'Potential', 'Age']
cols_exist = [c for c in numeric_cols if c in df_fifa.columns]

print("\nDescriptive Statistics (FIFA 23):")
print(df_fifa[cols_exist].describe().round(2))

print("\nAttribute Types (FIFA 23):")
print(df_fifa.dtypes.value_counts())

print("\nMissing Values (FIFA 23 - key columns):")
print(df_fifa[cols_exist].isnull().sum())

print("\nMissing Values (Transfermarkt):")
print(df_transfer.isnull().sum())

def simplify_position(pos):
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos).split(',')[0].strip()
    if pos in ['ST', 'CF', 'LW', 'RW', 'LF', 'RF', 'LS', 'RS']:
        return 'Forward'
    elif pos in ['CAM', 'CM', 'CDM', 'LM', 'RM']:
        return 'Midfielder'
    elif pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Defender'
    elif pos == 'GK':
        return 'Goalkeeper'
    else:
        return 'Other'

df_fifa['Position'] = df_fifa['BestPosition'].apply(simplify_position)
print("\nPosition Distribution:")
print(df_fifa['Position'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 1 — Data Understanding', fontsize=16, fontweight='bold')

pos_counts = df_fifa['Position'].value_counts()
axes[0, 0].bar(pos_counts.index, pos_counts.values,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[0, 0].set_title('Player Position Distribution (FIFA 23)')
axes[0, 0].set_xlabel('Position')
axes[0, 0].set_ylabel('Count')

if 'Overall' in df_fifa.columns:
    axes[0, 1].hist(df_fifa['Overall'].dropna(), bins=30, color='#58A6FF', edgecolor='white')
    axes[0, 1].set_title('Overall Rating Distribution')
    axes[0, 1].set_xlabel('Rating')
    axes[0, 1].set_ylabel('Count')

if 'Age' in df_fifa.columns:
    axes[1, 0].hist(df_fifa['Age'].dropna(), bins=25, color='#BC80F0', edgecolor='white')
    axes[1, 0].set_title('Age Distribution')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')

if len(cols_exist) >= 3:
    corr = df_fifa[cols_exist].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('phase1_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: phase1_charts.png")
print("Phase 1 Done!")
