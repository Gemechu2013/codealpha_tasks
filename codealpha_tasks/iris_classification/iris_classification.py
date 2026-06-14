"""
Iris Flower Species Classification
===================================
CodeAlpha Data Science Internship Task
Author: Gemechu Ejeta Atomsa

Objective:
    Train and evaluate multiple ML classifiers to predict Iris flower species
    (Setosa, Versicolor, Virginica) from sepal and petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── 1. Load & Prepare Data ──────────────────────────────────────────────────
df = pd.read_csv('data/Iris.csv')
df.drop(columns=['Id'], inplace=True)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['Species'].value_counts())
print("\nFirst 5 rows:\n", df.head())

X = df.drop(columns=['Species'])
y = df['Species']

le = LabelEncoder()
y_enc = le.fit_transform(y)  # setosa=0, versicolor=1, virginica=2

# ── 2. Train/Test Split & Scaling ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. Train Multiple Classifiers ───────────────────────────────────────────
models = {
    'Logistic Regression':   LogisticRegression(max_iter=200),
    'K-Nearest Neighbors':   KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True),
    'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n── Model Performance ──────────────────────────────────")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    results[name] = {'model': model, 'preds': preds, 'acc': acc, 'cv': cv}
    print(f"{name:28s} | Test Acc: {acc:.4f} | CV Acc: {cv:.4f}")

# ── 4. Best Model Report ────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['acc'])
best      = results[best_name]
print(f"\nBest model: {best_name} ({best['acc']*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, best['preds'], target_names=le.classes_))

# ── 5. Feature Importances ──────────────────────────────────────────────────
rf = results['Random Forest']['model']
feat_imp = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("Feature Importances (Random Forest):")
print(feat_imp)

# ── 6. Visualizations ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Iris Classification — Model Analysis', fontsize=16, fontweight='bold', y=1.01)

# (A) Model comparison
ax = axes[0, 0]
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
cvs    = [results[n]['cv']  for n in names]
x      = np.arange(len(names))
w      = 0.35
bars1  = ax.bar(x - w/2, accs, w, label='Test Accuracy', color='#4C72B0')
bars2  = ax.bar(x + w/2, cvs,  w, label='CV Accuracy',   color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
ax.legend(fontsize=8)
for b in bars1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for b in bars2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7)

# (B) Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.5)
ax.set_title(f'Confusion Matrix — {best_name}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# (C) Feature importances
ax = axes[1, 0]
feat_imp.plot(kind='bar', ax=ax,
              color=['#2ca02c', '#98df8a', '#ff7f0e', '#ffbb78'])
ax.set_title('Feature Importances (Random Forest)')
ax.set_ylabel('Importance')
ax.set_xticklabels(feat_imp.index, rotation=30, ha='right')
for i, v in enumerate(feat_imp):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

# (D) Petal scatter by species
ax = axes[1, 1]
colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
for cls in [0, 1, 2]:
    mask = y_enc == cls
    ax.scatter(X.loc[mask, 'PetalLengthCm'], X.loc[mask, 'PetalWidthCm'],
               c=colors[cls], label=le.classes_[cls],
               alpha=0.7, edgecolors='k', linewidths=0.4)
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Petal Dimensions by Species')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/iris_classification.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/iris_classification.png")
