import pandas as pd
import numpy as np
# Load the three CSV files
df1 = pd.read_csv(r"C:\Users\Hp\OneDrive - Luleå University of Technology\Master\LP3\Introduction to Industrial AI and eMaintenance D7015B 35654\Assignment 3\Trail1_extracted_features_acceleration_m1ai1-1.csv")
df2 = pd.read_csv(r"C:\Users\Hp\OneDrive - Luleå University of Technology\Master\LP3\Introduction to Industrial AI and eMaintenance D7015B 35654\Assignment 3\Trail2_extracted_features_acceleration_m1ai1.csv")
df3 = pd.read_csv(r"C:\Users\Hp\OneDrive - Luleå University of Technology\Master\LP3\Introduction to Industrial AI and eMaintenance D7015B 35654\Assignment 3\Trail3_extracted_features_acceleration_m2ai0.csv")
# Combine into a single dataset
combined_df = pd.concat([df1, df2, df3], ignore_index=True)
# Remove specified columns
columns_to_remove = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
combined_df = combined_df.drop(columns=[col for col in columns_to_remove if col in combined_df.columns])
# Replace 'normal' with 0 and all other events with 1
combined_df['event'] = combined_df['event'].apply(lambda x: 0 if x == 'normal' else 1)
#Verfication code
print("Shape:", combined_df.shape)
print("Columns:", combined_df.columns.tolist())
print("Event counts:\n", combined_df['event'].value_counts())
print(combined_df.head())
print(combined_df.info())
# STEP 2: Normalize the dataset (Z-score standardization)
# Assumes 'combined_df' from Step 1 exists with 17 columns (16 features + 'event')
# Identify feature columns (exclude target 'event')
feature_cols = combined_df.columns.drop('event')
# Apply Z-score normalization to features only: (x - mean) / std
normalized_df = combined_df.copy()
normalized_df[feature_cols] = (combined_df[feature_cols] - combined_df[feature_cols].mean()) / combined_df[feature_cols].std()
# Verify normalization worked (should show means ~0, stds ~1)
print("Shape:", normalized_df.shape)
print("Features normalized:", len(feature_cols))
print("\nSample verification (first feature 'mean'):")
print("Original mean:", combined_df['mean'].mean())
print("Normalized mean:", normalized_df['mean'].mean())
print("Normalized std:", normalized_df['mean'].std())
print("\nEvent distribution unchanged:")
print(normalized_df['event'].value_counts())
# STEP 3: Dataset splitting, Cross-Validation, SVM Comparison
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# 1. DATASET SPLITTING (80/20)
X = normalized_df.drop('event', axis=1)  # Features
y = normalized_df['event']               # Target
# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("=== 80/20 TRAIN-TEST SPLIT ===")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Test class distribution:\n{y_test.value_counts(normalize=True).round(3)}")
print()
# 2. TRAIN SVM ON 80/20 SPLIT
svm_80_20 = SVC(kernel='rbf', random_state=42)
svm_80_20.fit(X_train, y_train)
y_pred_80_20 = svm_80_20.predict(X_test)
acc_80_20 = accuracy_score(y_test, y_pred_80_20)
print(f"SVM 80/20 Test Accuracy: {acc_80_20:.4f}")
print("\nClassification Report (80/20):")
print(classification_report(y_test, y_pred_80_20))
print()
# 3. K-FOLD CROSS-VALIDATION (5-fold) on TRAINING set only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(SVC(kernel='rbf', random_state=42), X_train, y_train, cv=skf)
print("=== 5-FOLD CROSS-VALIDATION ON TRAINING SET ===")
print(f"5-fold CV scores: {kfold_scores.round(4)}")
print(f"CV Mean accuracy:  {kfold_scores.mean():.4f} (±{kfold_scores.std()*2:.4f})")
print()
# 4. COMPARISON TABLE
print("=== COMPARISON: 80/20 vs K-FOLD CV ===")
print(f"{'Method':<20} {'Accuracy':<10} {'Consistency'}")
print(f"{'80/20 Test Split':<20} {acc_80_20:.4f}{'':<10} Single run")
print(f"{'5-Fold CV (Train)':<20} {kfold_scores.mean():.4f}{'':<10} ±{kfold_scores.std():.4f}")
# FEATURE SELECTION: 4 Methods (Filter x2, Wrapper, Embedded)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
print("=== ORIGINAL TRAINING DATA ===")
print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")
print()
# ============================================================================
# 1. FILTER METHOD: Variance Threshold
# ============================================================================
print("1. FILTER: Variance Threshold (threshold=0.01)")
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X_train)
selected_var = X_train.columns[selector_var.get_support()].tolist()
print(f"  Retained: {len(selected_var)}/{X_train.shape[1]} features")
print(f"  Removed low-variance features")
print()
# ============================================================================
# 2. FILTER METHOD: SelectKBest with Chi-Square (top 8 features)
# ============================================================================
print("2. FILTER: SelectKBest Chi-Square (top 8 features)")
# Ensure non-negative data for chi2 (add small constant if needed)
X_train_chi2 = X_train - X_train.min() + 1e-6
selector_chi2 = SelectKBest(score_func=chi2, k=8)
X_chi2 = selector_chi2.fit_transform(X_train_chi2, y_train)
selected_chi2 = X_train.columns[selector_chi2.get_support()].tolist()
print(f"  Retained: {len(selected_chi2)} features")
print(f"  Features: {selected_chi2}")
print()
# ============================================================================
# 3. WRAPPER METHOD: Recursive Feature Elimination (RFE) with SVM
# ============================================================================
print("3. WRAPPER: RFE with SVM (top 8 features)")
svm_rfe = SVC(kernel="linear")
selector_rfe = RFE(estimator=svm_rfe, n_features_to_select=8, step=1)
X_rfe = selector_rfe.fit_transform(X_train, y_train)
selected_rfe = X_train.columns[selector_rfe.support_].tolist()
print(f"  Retained: {len(selected_rfe)} features")
print(f"  Features: {selected_rfe}")
print()
# ============================================================================
# 4. EMBEDDED METHOD: Tree-based Feature Importance (Random Forest)
# ============================================================================
print("4. EMBEDDED: RandomForest Feature Importance (top 8)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:8]
selected_rf = X_train.columns[indices].tolist()
print(f"  Retained: {len(selected_rf)} features")
print(f"  Features: {selected_rf}")
print("  Top 3:", 
      [f"{name}: {imp:.3f}" for name, imp in sorted(zip(X_train.columns, importances), 
                                                   key=lambda x: x[1], reverse=True)[:3]])
print()
# ============================================================================
# COMPARISON SUMMARY TABLE
# ============================================================================
print("=== FEATURE SELECTION RESULTS COMPARISON ===")
results = {
    'Variance Threshold': len(selected_var),
    'Chi-Square SelectKBest': len(selected_chi2),
    'RFE (SVM)': len(selected_rfe),
    'Random Forest Importance': len(selected_rf)
}
comparison_df = pd.DataFrame([
    ['Variance Threshold', len(selected_var), ', '.join(selected_var[:3])],
    ['Chi-Square SelectKBest', len(selected_chi2), ', '.join(selected_chi2[:3])],
    ['RFE (SVM)', len(selected_rfe), ', '.join(selected_rfe[:3])],
    ['Random Forest Importance', len(selected_rf), ', '.join(selected_rf[:3])]
], columns=['Method', 'Features Retained', 'Sample Selected Features'])
print(comparison_df.to_string(index=False))
print()
# ============================================================================
# SAVE ALL FEATURE SETS FOR MODEL COMPARISON
# ============================================================================
feature_sets = {
    'variance': selected_var,
    'chi2_kbest': selected_chi2,
    'rfe_svm': selected_rfe,
    'rf_importance': selected_rf
}
print("✅ FEATURE SETS SAVED:")
for name, features in feature_sets.items():
    print(f"  {name}: {len(features)} features")
print("\nUse feature_sets['method_name'] for model training comparison!")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Safety check: show available feature sets
print("Available feature sets:", list(feature_sets.keys()))
print()
results = []
# Baseline: ALL FEATURES (no feature selection)
print("=== BASELINE: SVM with ALL 16 FEATURES ===")
svm_all = SVC(kernel='rbf', random_state=42)
svm_all.fit(X_train, y_train)
y_pred_all = svm_all.predict(X_test)
acc_all = accuracy_score(y_test, y_pred_all)
print(f"Accuracy (all features): {acc_all:.4f}")
print()
results.append(['All features', X_train.shape[1], acc_all])
# LOOP OVER FEATURE SETS
for name, feat_list in feature_sets.items():
    print(f"=== SVM with FEATURE SET: {name} ===")
    print(f"Using {len(feat_list)} features: {feat_list}")
    # Subset train/test using selected features
    X_train_fs = X_train[feat_list]
    X_test_fs = X_test[feat_list]
    # Train SVM
    svm_fs = SVC(kernel='rbf', random_state=42)
    svm_fs.fit(X_train_fs, y_train)
    y_pred_fs = svm_fs.predict(X_test_fs)
    # Evaluate
    acc_fs = accuracy_score(y_test, y_pred_fs)
    print(f"Accuracy ({name}): {acc_fs:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_fs, digits=3))
    print()
    # Save result
    results.append([name, len(feat_list), acc_fs])
# SUMMARY COMPARISON TABLE
print("=== SVM PERFORMANCE COMPARISON ===")
results_df = pd.DataFrame(results, columns=['Feature Set', 'Num Features', 'Test Accuracy'])
# Sort by accuracy descending
results_df = results_df.sort_values(by='Test Accuracy', ascending=False)
print(results_df.to_string(index=False))
best_row = results_df.iloc[0]
print("\nBEST FEATURE SELECTION METHOD:")
print(f"- Feature set: {best_row['Feature Set']}")
print(f"- Num features: {best_row['Num Features']}")
print(f"- Test accuracy: {best_row['Test Accuracy']:.4f}")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# CREATE importance_df FIRST (this was missing!)
# Using the trained Random Forest from feature selection (rf variable exists)
importances = rf.feature_importances_
feature_names = X_train.columns
# Create importance DataFrame and sort (FIX FOR ERROR)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
# Set style for publication-ready plots
plt.style.use('default')
sns.set_palette("husl")
print("=== FEATURE IMPORTANCE VISUALIZATIONS (Railway Switch Anomaly Detection) ===")

# =============================================================================
# PLOT 1: TOP 10 Feature Importance (Horizontal bar - BEST FOR PRESENTATION)
# =============================================================================
plt.figure(figsize=(12, 8))
top10 = importance_df.head(10)
bars = plt.barh(range(len(top10)), top10['importance'], color='steelblue', alpha=0.8, edgecolor='navy')
plt.yticks(range(len(top10)), top10['feature'])
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.title('Top 10 Feature Importance (Random Forest)\nRailway Switch Acceleration Anomalies', fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()  # Highest at top
plt.grid(axis='x', alpha=0.3)
# Add value labels on bars
for i, (bar, imp) in enumerate(zip(bars, top10['importance'])):
    plt.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
# =============================================================================
# PLOT 2: All 16 Features (Vertical bar - Complete ranking)
# =============================================================================
plt.figure(figsize=(14, 8))
colors = plt.cm.Blues(np.linspace(0.3, 1, len(importance_df)))
bars = plt.bar(range(len(importance_df)), importance_df['importance'], color=colors, alpha=0.8, edgecolor='darkblue')
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Importance Score', fontsize=14, fontweight='bold')
plt.title('All 16 Feature Importances (Random Forest)\nAcceleration Domain Features', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3)
# Add value labels
for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'])):
    plt.text(i, imp + 0.01, f'{imp:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()
# =============================================================================
# PLOT 3: Feature Selection Methods - Features Retained
# =============================================================================
plt.figure(figsize=(10, 6))
methods = ['Variance\nThreshold', 'Chi²\nSelectKBest', 'RFE\n(SVM)', 'RF\nImportance']
features_kept = [len(feature_sets['variance']), len(feature_sets['chi2_kbest']), 
                len(feature_sets['rfe_svm']), len(feature_sets['rf_importance'])]
colors = ['#2E8B57', '#FF8C00', '#8A2BE2', '#DC143C']
bars = plt.bar(methods, features_kept, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylabel('Features Retained', fontsize=14, fontweight='bold')
plt.title('Features Retained by Each Selection Method\n(Original: 16 features)', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, max(features_kept)+2)
plt.grid(axis='y', alpha=0.3)
# Add value labels
for i, (bar, v) in enumerate(zip(bars, features_kept)):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v), ha='center', va='bottom', 
             fontsize=14, fontweight='bold', color='white')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# =============================================================================
# PLOT 4: SVM Accuracy vs Feature Selection Method
# =============================================================================
plt.figure(figsize=(11, 7))
feature_sets_names = ['All Features', 'Variance', 'Chi² KBest', 'RFE-SVM', 'RF Importance']
accuracies = [acc_all, results[1][2], results[2][2], results[3][2], results[4][2]]
colors_acc = ['#4682B4', '#2E8B57', '#FF8C00', '#8A2BE2', '#DC143C']
bars = plt.bar(feature_sets_names, accuracies, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
plt.title('SVM Test Accuracy by Feature Selection Method\nRBF Kernel, 80/20 Split (30 Test Samples)', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0.90, 1.02)
plt.grid(axis='y', alpha=0.3)
# Add value labels and best marker
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f'{acc:.1%}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    if acc == max(accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, acc - 0.025, '🏆 BEST', 
                ha='center', va='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

