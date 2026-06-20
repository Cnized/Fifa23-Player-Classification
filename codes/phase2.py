import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

os.chdir(r"C:\Users\Kian K\Downloads\داده کاوی\Project")

df_fifa = pd.read_csv("data/players_fifa23.csv", low_memory=False)

leagues = ['laliga', 'premier_league', 'ligue1', 'bundesliga', 'serie_a']
base_path = "data/transfermarkt-data-master"
all_transfers = []
for league in leagues:
    league_path = os.path.join(base_path, league)
    if not os.path.exists(league_path):
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

fifa_before = df_fifa.shape[0]
transfer_before = df_transfer.shape[0]

missing_before_fifa = df_transfer[['market_value', 'fee', 'dealing_country', 'age', 'nationality']].isnull().sum()
missing_before_transfer = df_transfer[['market_value', 'fee']].copy()

fifa_cols = ['Name', 'Age', 'Nationality', 'BestPosition', 'Overall', 'Potential',
             'PaceTotal', 'ShootingTotal', 'PassingTotal', 'DribblingTotal',
             'DefendingTotal', 'PhysicalityTotal', 'ValueEUR', 'WageEUR',
             'PreferredFoot', 'WeakFoot', 'SkillMoves', 'IntReputation']
df_fifa = df_fifa[[c for c in fifa_cols if c in df_fifa.columns]].copy()

transfer_cols = ['player_name', 'age', 'nationality', 'position', 'pos',
                 'market_value', 'fee', 'is_loan', 'league', 'window', 'movement']
df_transfer = df_transfer[[c for c in transfer_cols if c in df_transfer.columns]].copy()

market_before = missing_before_transfer['market_value'].copy()
fee_before = missing_before_transfer['fee'].copy()

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
df_fifa = df_fifa[~df_fifa['Position'].isin(['Unknown', 'Other'])].copy()

df_transfer['market_value'] = df_transfer['market_value'].fillna(df_transfer['market_value'].median())
df_transfer['fee'] = df_transfer['fee'].fillna(0)
df_transfer = df_transfer.dropna(subset=['age', 'nationality'])
df_transfer = df_transfer[df_transfer['movement'] == 'in'].copy()

numeric_cols = ['PaceTotal', 'ShootingTotal', 'PassingTotal', 'DribblingTotal',
                'DefendingTotal', 'PhysicalityTotal', 'Overall', 'Potential', 'Age']
cols_to_normalize = [c for c in numeric_cols if c in df_fifa.columns]

for col in cols_to_normalize:
    df_fifa[col] = df_fifa[col].fillna(df_fifa[col].median())

df_fifa_before_norm = df_fifa[cols_to_normalize].copy()
scaler = MinMaxScaler()
df_fifa[cols_to_normalize] = scaler.fit_transform(df_fifa[cols_to_normalize])

fifa_after = df_fifa.shape[0]
transfer_after = df_transfer.shape[0]

print("=" * 50)
print("Before/After Summary:")
print(f"  FIFA 23:       {fifa_before} --> {fifa_after} records")
print(f"  Transfermarkt: {transfer_before} --> {transfer_after} records")
print("\nMissing Values Fixed (Transfermarkt):")
print(missing_before_fifa)
print("\nPosition Distribution:")
print(df_fifa['Position'].value_counts())

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Phase 2 — Data Preprocessing', fontsize=16, fontweight='bold')

missing_data = pd.Series({
    'market_value': 25129,
    'fee': 11778,
    'dealing_country': 3160,
    'age': 5,
    'nationality': 15
})
axes[0, 0].bar(missing_data.index, missing_data.values, color='#FF6B6B', edgecolor='white')
axes[0, 0].set_title('Missing Values Before Cleaning')
axes[0, 0].set_xlabel('Column')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=30)

summary = pd.DataFrame({
    'Before': [fifa_before, transfer_before],
    'After': [fifa_after, transfer_after]
}, index=['FIFA 23', 'Transfermarkt'])
x = np.arange(len(summary))
w = 0.35
axes[0, 1].bar(x - w/2, summary['Before'], w, label='Before', color='#FF6B6B')
axes[0, 1].bar(x + w/2, summary['After'], w, label='After', color='#4ECDC4')
axes[0, 1].set_title('Records Before vs After Preprocessing')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(summary.index)
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

pos_counts = df_fifa['Position'].value_counts()
axes[0, 2].bar(pos_counts.index, pos_counts.values,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 2].set_title('Position Distribution (After Cleaning)')
axes[0, 2].set_xlabel('Position')
axes[0, 2].set_ylabel('Count')

axes[1, 0].boxplot([df_fifa_before_norm[c].dropna() for c in cols_to_normalize[:5]],
                   labels=cols_to_normalize[:5])
axes[1, 0].set_title('Feature Distribution BEFORE Normalization')
axes[1, 0].set_ylabel('Value')
axes[1, 0].tick_params(axis='x', rotation=30)

axes[1, 1].boxplot([df_fifa[c] for c in cols_to_normalize[:5]],
                   labels=cols_to_normalize[:5])
axes[1, 1].set_title('Feature Distribution AFTER Normalization (0-1)')
axes[1, 1].set_ylabel('Value')
axes[1, 1].tick_params(axis='x', rotation=30)

transfer_log = np.log1p(df_transfer['market_value'])
axes[1, 2].hist(transfer_log, bins=40, color='#58A6FF', edgecolor='white')
axes[1, 2].set_title('Market Value Distribution (log scale)')
axes[1, 2].set_xlabel('log(Market Value)')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('phase2_charts.png', dpi=150, bbox_inches='tight')
plt.show()

df_fifa.to_csv('data/fifa_cleaned.csv', index=False)
df_transfer.to_csv('data/transfer_cleaned.csv', index=False)

print("\nSaved: data/fifa_cleaned.csv")
print("Saved: data/transfer_cleaned.csv")
print("Phase 2 Done!")
