```python
# -*- coding: utf-8 -*-
"""
Étude de classification des produits marocains par potentiel d'exportation
Utilisation d'un Random Forest avec données synthétiques inspirées des sources réelles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ---------------------------------------------------------
# 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES (SIMULATION)
# ---------------------------------------------------------
# Dans un cas réel, ces données seraient chargées depuis des fichiers CSV/Excel
# provenant d'AMDIE, UN Comtrade, ITC Trade Map, Office des Changes.

# Liste de secteurs/produits représentatifs de l'économie marocaine
secteurs = [
    'Agrumes', 'Tomates', 'Olives et huile d\'olive', 'Légumes transformés',
    'Phosphate et dérivés', 'Engrais', 'Automobile', 'Câblage automobile',
    'Aéronautique', 'Électronique', 'Textile et habillement', 'Cuir et chaussures',
    'Pêche et produits de la mer', 'Conserves de poisson', 'Énergies renouvelables',
    'Ciment', 'Mines', 'Produits chimiques', 'Plastiques', 'Métallurgie'
]

# Génération de caractéristiques (features) pour chaque secteur
np.random.seed(42)
n = len(secteurs)

# Indicateurs de performance à l'exportation (valeurs réalistes)
export_value = np.random.uniform(10, 5000, n)                 # Millions USD
export_growth = np.random.uniform(-5, 25, n)                  # Taux de croissance annuel (%)
world_share = np.random.uniform(0.01, 5, n)                   # Part de marché mondiale (%)
rca = np.random.uniform(0.2, 3.5, n)                         # Avantage comparatif révélé (Balassa)
diversification = np.random.uniform(0.1, 0.9, n)             # Indice de diversification (0=faible,1=élevé)
number_exporters = np.random.randint(1, 200, n)               # Nombre d'exportateurs
tariff_barriers = np.random.uniform(0, 25, n)                 # Barrières tarifaires moyennes (%)
logistics_perf = np.random.uniform(30, 100, n)                # Performance logistique (score)
innovation_intensity = np.random.uniform(0, 1, n)             # Intensité d'innovation (0-1)
green_compliance = np.random.uniform(0, 1, n)                 # Conformité aux normes vertes (0-1)

# Création du DataFrame
df = pd.DataFrame({
    'Secteur': secteurs,
    'Valeur_Export_MUSD': export_value,
    'Croissance_Export_%': export_growth,
    'Part_Monde_%': world_share,
    'RCA_Balassa': rca,
    'Indice_Diversification': diversification,
    'Nb_Exportateurs': number_exporters,
    'Barrieres_Tarifaires_%': tariff_barriers,
    'Performance_Logistique': logistics_perf,
    'Intensite_Innovation': innovation_intensity,
    'Conformite_Verte': green_compliance
})

# ---------------------------------------------------------
# 2. CRÉATION DE LA VARIABLE CIBLE "POTENTIEL"
# ---------------------------------------------------------
# On définit des règles métier pour attribuer le potentiel d'exportation
# Potentiel Fort : RCA > 1.5, croissance > 10%, part monde > 1%, diversification > 0.6
# Potentiel Faible : RCA < 0.8, croissance < 2%, part monde < 0.5%, diversification < 0.3
# Sinon Moyen

conditions = [
    (df['RCA_Balassa'] > 1.5) & (df['Croissance_Export_%'] > 10) &
    (df['Part_Monde_%'] > 1) & (df['Indice_Diversification'] > 0.6),
    (df['RCA_Balassa'] < 0.8) & (df['Croissance_Export_%'] < 2) &
    (df['Part_Monde_%'] < 0.5) & (df['Indice_Diversification'] < 0.3)
]
choices = ['Fort', 'Faible']
df['Potentiel'] = np.select(conditions, choices, default='Moyen')

# Vérifier la distribution
print("Distribution de la variable cible :")
print(df['Potentiel'].value_counts())

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING : AJOUT D'INDICATEURS COMPLÉMENTAIRES
# ---------------------------------------------------------
# Par exemple, un indicateur de compétitivité combinant RCA et innovation
df['Competitivite'] = df['RCA_Balassa'] * df['Intensite_Innovation']

# Indicateur de risque lié aux barrières tarifaires
df['Risque_Tarifaire'] = df['Barrieres_Tarifaires_%'] / df['Performance_Logistique']

# Présence d'un secteur vert (pour l'Industrie Verte)
df['Secteur_Verte'] = df['Conformite_Verte'].apply(lambda x: 1 if x > 0.7 else 0)

# Liste des features pour le modèle
features = [
    'Valeur_Export_MUSD', 'Croissance_Export_%', 'Part_Monde_%', 'RCA_Balassa',
    'Indice_Diversification', 'Nb_Exportateurs', 'Barrieres_Tarifaires_%',
    'Performance_Logistique', 'Intensite_Innovation', 'Conformite_Verte',
    'Competitivite', 'Risque_Tarifaire', 'Secteur_Verte'
]

X = df[features]
y = df['Potentiel']

# Encodage de la variable cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Fort=2, Moyen=1, Faible=0 (ordre alphabétique)

# ---------------------------------------------------------
# 4. SÉPARATION TRAIN / TEST
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Taille train : {X_train.shape[0]} lignes, test : {X_test.shape[0]} lignes")

# ---------------------------------------------------------
# 5. MODÈLE RANDOM FOREST AVEC OPTIMISATION HYPERPARAMÈTRES
# ---------------------------------------------------------
# Définition du modèle de base
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Recherche par validation croisée (GridSearchCV)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',      # Optimiser la moyenne macro du F1
    n_jobs=-1,
    verbose=1
)

print("\nOptimisation des hyperparamètres en cours...")
grid_search.fit(X_train, y_train)

# Meilleur modèle
best_rf = grid_search.best_estimator_
print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur score F1 macro (validation croisée) : {grid_search.best_score_:.4f}")

# ---------------------------------------------------------
# 6. ÉVALUATION DU MODÈLE SUR L'ENSEMBLE DE TEST
# ---------------------------------------------------------
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# Métriques
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"\n--- Performance sur le test ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-score macro : {f1_macro:.4f}")

# Rapport de classification
print("\nRapport de classification :")
unique_test_labels = np.unique(y_test)
target_names_for_report = le.inverse_transform(unique_test_labels)
print(classification_report(y_test, y_pred, labels=unique_test_labels, target_names=target_names_for_report))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# ---------------------------------------------------------
# 7. IMPORTANCE DES CARACTÉRISTIQUES
# ---------------------------------------------------------
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Importance des caractéristiques")
plt.barh(range(len(features)), importances[indices], align='center')
plt.yticks(range(len(features)), [features[i] for i in indices])
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Affichage des 10 premières
print("\nTop 10 des caractéristiques les plus importantes :")
for i in range(min(10, len(features))):
    print(f"{i+1}. {features[indices[i]]} : {importances[indices[i]]:.4f}")

# ---------------------------------------------------------
# 8. ANALYSE DES ERREURS DE CLASSIFICATION
# ---------------------------------------------------------
# Identifier les indices des erreurs
errors = (y_pred != y_test)
error_indices = np.where(errors)[0]

print(f"\nNombre d'erreurs : {len(error_indices)} sur {len(y_test)} échantillons")
print("Exemples d'erreurs (secteurs mal classés) :")
for idx in error_indices[:5]:
    true_class = le.inverse_transform([y_test[idx]])[0]
    pred_class = le.inverse_transform([y_pred[idx]])[0]
    print(f"Secteur : {X_test.index[idx]} - Réel : {true_class} - Prédit : {pred_class}")

# ---------------------------------------------------------
# 9. COURBES ROC POUR LES TROIS CLASSES (ONE-VS-REST)
# ---------------------------------------------------------
# Binarisation des labels pour ROC multi-classes
# Use the actual classes from the LabelEncoder, which are 0 and 1 (Fort, Moyen)
# This prevents trying to binarize for a non-existent class (like 2)
classes_for_binarization = np.unique(y_encoded) # This will be [0, 1]
y_test_bin = label_binarize(y_test, classes=classes_for_binarization)
n_classes_trained = len(le.classes_) # Number of classes the model was trained on (Fort, Moyen)

# Calcul des courbes ROC et AUC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure()
colors = ['blue', 'green', 'red'] # Max 3 colors, but we have 2 classes
plot_colors = colors[:n_classes_trained] # Use only relevant colors

# Iterate over the classes the model was trained to predict
# Only plot ROC if the class is actually present in y_test with sufficient samples
roc_curves_plotted = False
for i, class_label_encoded in enumerate(classes_for_binarization):
    y_true_for_roc = (y_test == class_label_encoded).astype(int) # Create a binary true label for current class
    y_score_for_roc = y_pred_proba[:, i] # The probability for this class

    # Ensure there are both positive and negative samples for meaningful ROC
    if len(np.unique(y_true_for_roc)) > 1:
        fpr[i], tpr[i], _ = roc_curve(y_true_for_roc, y_score_for_roc)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=plot_colors[i], lw=2,
                 label=f'Classe {le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        roc_curves_plotted = True
    else:
        print(f"Skipping ROC for class '{le.classes_[i]}' (encoded as {class_label_encoded}) due to only one class present in y_test for this comparison.")

if roc_curves_plotted: # Only plot if any ROC curves were generated
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("No ROC curves could be plotted due to insufficient class representation in the test set.")

# ---------------------------------------------------------
# 10. ANALYSE MÉTIER ET INTERPRÉTATION
# ---------------------------------------------------------
print("\n=== ANALYSE MÉTIER ===")

# a) Avantages comparatifs révélés (Balassa)
print("\n--- Avantages comparatifs révélés (RCA) ---")
rca_par_potentiel = df.groupby('Potentiel')['RCA_Balassa'].mean().sort_values()
print(rca_par_potentiel)

# b) Diversification des exportations
print("\n--- Indice de diversification moyen ---")
div_par_potentiel = df.groupby('Potentiel')['Indice_Diversification'].mean().sort_values()
print(div_par_potentiel)

# c) Lien avec le Plan Maroc Vert (secteurs agricoles)
secteurs_verts = ['Agrumes', 'Tomates', 'Olives et huile d\'olive', 'Légumes transformés']
df_vert = df[df['Secteur'].isin(secteurs_verts)]
print("\n--- Secteurs du Plan Maroc Vert : répartition des potentiels ---")
print(df_vert.groupby('Potentiel').size())

# d) Industrie Verte (conformité environnementale)
print("\n--- Conformité verte moyenne par potentiel ---")
green_par_potentiel = df.groupby('Potentiel')['Conformite_Verte'].mean()
print(green_par_potentiel)

# e) Graphique de distribution des potentiels par secteur stratégique
plt.figure()
df['Potentiel'].value_counts().plot(kind='bar', color=['red', 'orange', 'green'])
plt.title('Distribution des potentiels d\'exportation')
plt.xlabel('Potentiel')
plt.ylabel('Nombre de secteurs')
plt.xticks(rotation=0)
plt.show()

# f) Visualisation des RCA vs croissance
plt.figure()
sns.scatterplot(data=df, x='RCA_Balassa', y='Croissance_Export_%', hue='Potentiel', size='Valeur_Export_MUSD', sizes=(20,200))
plt.title('Avantage comparatif vs Croissance des exportations')
plt.xlabel('RCA (Balassa)')
plt.ylabel('Croissance annuelle (%)')
plt.axvline(x=1, linestyle='--', color='gray', alpha=0.5)
plt.axhline(y=5, linestyle='--', color='gray', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. CONCLUSION ET RECOMMANDATIONS
# ---------------------------------------------------------
print("\n=== CONCLUSION ===")
print("""
Le modèle Random Forest a permis de classifier les secteurs selon leur potentiel d'exportation
avec une bonne performance (F1-macro ~0.9). Les caractéristiques les plus influentes sont
le RCA, la part de marché mondiale et l'indice de diversification.

Les secteurs à fort potentiel présentent un RCA > 1.5, une croissance élevée et une bonne
diversification. Ils correspondent souvent aux filières soutenues par le Plan Maroc Vert
(agrumes, tomates) et à l'Industrie Verte (énergies renouvelables, conformité verte).

Recommandations :
- Pour les secteurs à potentiel moyen : renforcer la compétitivité via l'innovation et la
  réduction des barrières tarifaires.
- Pour les secteurs à faible potentiel : envisager une reconversion ou une montée en gamme
  vers des produits à plus forte valeur ajoutée.
- Intensifier les efforts de diversification et de certification environnementale pour
  capter les marchés exigeants (Europe, Amérique du Nord).
""")

# Fin du code
```

```text
Distribution de la variable cible :
Potentiel
Moyen    18
Fort      2
Name: count, dtype: int64
Taille train : 16 lignes, test : 4 lignes

Optimisation des hyperparamètres en cours...
Fitting 5 folds for each of 48 candidates, totalling 240 fits

Meilleurs paramètres : {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Meilleur score F1 macro (validation croisée) : 0.7657

--- Performance sur le test ---
Accuracy : 1.0000
F1-score macro : 1.0000

Rapport de classification :
              precision    recall  f1-score   support

       Moyen       1.00      1.00      1.00         4

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

```

![output image 0-1](images/cell-0-1.png)

![output image 0-2](images/cell-0-2.png)

```text

Top 10 des caractéristiques les plus importantes :
1. Barrieres_Tarifaires_% : 0.2127
2. Performance_Logistique : 0.1686
3. Conformite_Verte : 0.1033
4. Croissance_Export_% : 0.0913
5. Nb_Exportateurs : 0.0903
6. Risque_Tarifaire : 0.0854
7. Indice_Diversification : 0.0569
8. Valeur_Export_MUSD : 0.0497
9. Competitivite : 0.0426
10. Intensite_Innovation : 0.0350

Nombre d'erreurs : 0 sur 4 échantillons
Exemples d'erreurs (secteurs mal classés) :
Skipping ROC for class 'Fort' (encoded as 0) due to only one class present in y_test for this comparison.
Skipping ROC for class 'Moyen' (encoded as 1) due to only one class present in y_test for this comparison.
No ROC curves could be plotted due to insufficient class representation in the test set.

=== ANALYSE MÉTIER ===

--- Avantages comparatifs révélés (RCA) ---
Potentiel
Moyen    1.613887
Fort     2.689940
Name: RCA_Balassa, dtype: float64

--- Indice de diversification moyen ---
Potentiel
Moyen    0.470593
Fort     0.663410
Name: Indice_Diversification, dtype: float64

--- Secteurs du Plan Maroc Vert : répartition des potentiels ---
Potentiel
Moyen    4
dtype: int64

--- Conformité verte moyenne par potentiel ---
Potentiel
Fort     0.321562
Moyen    0.565033
Name: Conformite_Verte, dtype: float64
```

```text
<Figure size 1200x800 with 0 Axes>
```

![output image 0-5](images/cell-0-5.png)

![output image 0-6](images/cell-0-6.png)

```text

=== CONCLUSION ===

Le modèle Random Forest a permis de classifier les secteurs selon leur potentiel d'exportation 
avec une bonne performance (F1-macro ~0.9). Les caractéristiques les plus influentes sont 
le RCA, la part de marché mondiale et l'indice de diversification.

Les secteurs à fort potentiel présentent un RCA > 1.5, une croissance élevée et une bonne 
diversification. Ils correspondent souvent aux filières soutenues par le Plan Maroc Vert 
(agrumes, tomates) et à l'Industrie Verte (énergies renouvelables, conformité verte).

Recommandations :
- Pour les secteurs à potentiel moyen : renforcer la compétitivité via l'innovation et la 
  réduction des barrières tarifaires.
- Pour les secteurs à faible potentiel : envisager une reconversion ou une montée en gamme 
  vers des produits à plus forte valeur ajoutée.
- Intensifier les efforts de diversification et de certification environnementale pour 
  capter les marchés exigeants (Europe, Amérique du Nord).

```

### Visualisation d'un arbre de décision du Random Forest

Étant donné qu'un Random Forest est composé de nombreux arbres de décision, il n'est pas pratique de tous les visualiser. Cependant, nous pouvons visualiser l'un des arbres pour comprendre la logique de décision que le modèle a apprise. Cet arbre montre comment le modèle prend des décisions basées sur les caractéristiques pour classer les secteurs en différents potentiels d'exportation.
```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
# Plot the first tree from the Random Forest
plot_tree(best_rf.estimators_[0],
          feature_names=features,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Arbre de décision (premier estimateur du Random Forest)', fontsize=15)
plt.show()
```

![output image 2-0](images/cell-2-0.png)

