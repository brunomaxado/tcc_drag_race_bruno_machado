import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


fim_posicao  = pd.read_excel('fim_posicao_queen.xlsx',  sheet_name=0)
meio_posicao = pd.read_excel('meio_posicao_queen.xlsx', sheet_name=0)

df = meio_posicao.copy()


features = ['bom', 'ruim', 'media', 'idade']
X = df[features]
y = df['vencedora']
groups = df['tempfranquia']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("Temporadas no treino:", df['tempfranquia'].iloc[train_idx].unique())
print("Temporadas no teste:",  df['tempfranquia'].iloc[test_idx].unique())


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_scaled, y_train)


y_pred_rf = rf.predict(X_test_scaled)
probs_rf  = rf.predict_proba(X_test_scaled)[:, 1]


print("\nRandom Forest Classifier")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
print("Matriz de Confusão:\n", cm)

roc_auc = roc_auc_score(y_test, probs_rf)
print("ROC-AUC:", roc_auc)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cbar=False)
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.tight_layout()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, probs_rf)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

queens_do_teste = meio_posicao.loc[X_test.index, 'queen']

resultados = pd.DataFrame({
    'queen': queens_do_teste,
    'Chance de Vencer (%)': (probs_rf * 100).round(2)
})

print("\n--- Chance de cada rainha ganhar (Random Forest) ---")
print(resultados.sort_values(by='Chance de Vencer (%)', ascending=False))

df_ranking = pd.DataFrame({
    "queen": df.loc[y_test.index, "queen"],
    "temporada": df.loc[y_test.index, "tempfranquia"],
    "real": y_test,
    "prob_vencer": probs_rf
}).sort_values(["temporada", "prob_vencer"], ascending=[True, False])

top_1 = df_ranking.groupby("temporada").head(1)
print("\nTop 1 por temporada (Random Forest):")
print(top_1)


importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nImportância das features:")
print(importancias)

plt.figure()
sns.barplot(x=importancias.values, y=importancias.index)
plt.title("Importância das Features - Random Forest")
plt.xlabel("Importância (Gini)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
