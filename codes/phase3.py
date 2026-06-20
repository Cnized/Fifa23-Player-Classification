import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

os.chdir(r"C:\Users\Kian K\Downloads\داده کاوی\Project")

df = pd.read_csv("data/fifa_cleaned.csv")

le = LabelEncoder()
df['Position_encoded'] = le.fit_transform(df['Position'])

feature_cols = ['PaceTotal', 'ShootingTotal', 'PassingTotal', 'DribblingTotal',
                'DefendingTotal', 'PhysicalityTotal', 'Overall', 'Potential', 'Age']

X = df[feature_cols]
y = df['Position_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} records")
print(f"Testing set:  {X_test.shape[0]} records")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))))
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("\n" + "=" * 50)
print("Model Accuracy Comparison:")
print(f"  Decision Tree:       {dt_acc:.4f} ({dt_acc*100:.2f}%)")
print(f"  KNN:                 {knn_acc:.4f} ({knn_acc*100:.2f}%)")
print(f"  Random Forest:       {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"  Logistic Regression: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

print("\nBest Model:", max([("Decision Tree", dt_acc), ("KNN", knn_acc),
                            ("Random Forest", rf_acc), ("Logistic Regression", lr_acc)],
                           key=lambda x: x[1])[0])

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

class_names = le.classes_

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Phase 3 — Classification Results', fontsize=16, fontweight='bold')

models = ['Decision Tree', 'KNN', 'Random Forest', 'Logistic Regression']
accuracies = [dt_acc, knn_acc, rf_acc, lr_acc]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = axes[0, 0].bar(models, accuracies, color=colors, edgecolor='white')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].tick_params(axis='x', rotation=15)
for bar, acc in zip(bars, accuracies):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

for ax, (pred, title) in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],
                               [(dt_pred, 'Decision Tree'), (knn_pred, 'KNN'),
                                (rf_pred, 'Random Forest'), (lr_pred, 'Logistic Regression')]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'Confusion Matrix — {title}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

feature_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
axes[1, 2].barh(feature_importance.index, feature_importance.values, color='#58A6FF')
axes[1, 2].set_title('Feature Importance (Random Forest)')
axes[1, 2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('phase3_charts.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: phase3_charts.png")
print("Phase 3 Done!")
