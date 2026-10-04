# Risques Sismiques

Ce projet utilise les données de l'USGS de l'année **2025** pour analyser les risques liés aux séismes et leurs impacts.

Les différents notebooks couvrent la préparation des données, l'entraînement et la comparaison des modèles, l'estimation des dommages, la recommandation d'actions et l'estimation des pertes économiques.

Le dataset principal contient **29 062 séismes** et **35 colonnes**.

## Données

**Source :** USGS Earthquake Catalog et USGS PAGER  
**Période :** 2025

Les principales variables utilisées sont :

- magnitude
- profondeur
- latitude
- longitude
- MMI
- CDI
- `felt`
- `sig`
- tsunami
- alerte PAGER

Toutes les informations ne sont pas disponibles pour tous les séismes. Le nombre d'observations utilisées varie donc selon le modèle.

---
## Modèle de risques des zones

**Notebook :** `02_Model_Risk_Zone.ipynb`

L'objectif est de prédire les zone selon niveau de risque d'un séisme.

Trois classes sont utilisées :

- Faible
- Moyen
- Élévé

Les classes sont basées sur les alertes l'USGS.

29062 séismes disposent des informations nécessaires.

| Classe | Nombre |
|---|---:|
| Faible | 10831 |
| Moyen | 18087 |
| Élévé | 144 |

Les données sont séparées en :

- Entraînement : **23249 séismes**
- Test : **5813 séismes**

### Modèles testés

| Modèle | Accuracy | Recall Sévère | F1 Sévère |
|---|---:|---:|---:|
| KNN | 96,1 % | 0,33 | 0,50 |
| Decision Tree | 98,3 % | 0,67 | 0,67 |
| Random Forest | 98,9 % | 0,67 | 0,80 |

---

## Modèle de dommages

**Notebook :** `05_Model_Damage.ipynb`

L'objectif est de prédire le niveau de dommages d'un séisme.

Trois classes sont utilisées :

- Faible
- Modéré
- Sévère

Les classes sont basées sur les alertes PAGER de l'USGS.

904 séismes disposent des informations nécessaires.

| Classe | Nombre |
|---|---:|
| Faible | 865 |
| Modéré | 25 |
| Sévère | 14 |

Le dataset est fortement déséquilibré.

Les données sont séparées en :

- Entraînement : **723 séismes**
- Test : **181 séismes**

### Modèles testés

| Modèle | Accuracy | Recall Sévère | F1 Sévère |
|---|---:|---:|---:|
| KNN | 96,1 % | 0,33 | 0,50 |
| Decision Tree | 98,3 % | 0,67 | 0,67 |
| Random Forest | 98,9 % | 0,67 | 0,80 |

### Champion

Le modèle retenu est **Random Forest**.

Accuracy sur le jeu de test : **98,9 %**

Résultats sur les données de test :

- Faible : **172 / 173**
- Modéré : **5 / 5**
- Sévère : **2 / 3**

Comme les classes sont déséquilibrées, l'accuracy n'est pas utilisée seule. Le Recall et le F1-score de la classe Sévère sont également pris en compte.

La validation croisée est utilisée pour vérifier la stabilité des résultats.

---

## Modèle d'action

**Notebook :** `06_Model_Action.ipynb`

L'objectif est de déterminer l'action à prendre selon le niveau de dommages.

| Niveau | Action | Priorité |
|---|---|---|
| Faible | Surveillance | Faible |
| Modéré | Évaluation prioritaire | Moyenne |
| Sévère | Intervention prioritaire | Élevée |

Ce notebook utilise des règles de décision. Il ne s'agit donc pas d'un modèle de Machine Learning avec un algorithme champion.

904 séismes ont été traités.

| Action | Nombre |
|---|---:|
| Surveillance | 865 |
| Évaluation prioritaire | 25 |
| Intervention prioritaire | 14 |

**107 alertes tsunami** ont également été identifiées.

Vérification finale :

- Actions manquantes : 0
- Actions indéterminées : 0
- Priorités manquantes : 0
- IDs dupliqués : 0
- Incohérences : 0

Le résultat est enregistré dans :

`data/action_model_2025.csv`

---

## Modèle de pertes économiques

**Notebook :** `07_Model_Economic_Loss.ipynb`

L'objectif est d'estimer les pertes économiques associées à un séisme.

Les informations économiques proviennent de USGS PAGER.

Le dataset final contient **734 séismes** :

- 625 avec une perte égale à zéro
- 109 avec une perte positive

Les variables utilisées sont :

- magnitude
- profondeur
- `sig`
- latitude
- longitude
- tsunami

La cible est transformée avec `log1p` car les pertes économiques sont très déséquilibrées.

### Modèles testés

- Linear Regression
- Random Forest
- Gradient Boosting

### Champion

Le modèle retenu est **Gradient Boosting**.

| Métrique | Résultat |
|---|---:|
| R² | 0,6859 |
| MAE | 353,47 M$ |
| RMSE | 4,27 G$ |

Pour les **22 séismes** du jeu de test ayant une perte économique positive :

- MAE : environ **2,36 G$**
- RMSE : environ **11,03 G$**

La validation croisée est également utilisée pour comparer les modèles.

Gradient Boosting obtient un **R² moyen d'environ 0,6675** en validation croisée.

---

## Validation et suivi

La **validation croisée (Cross-Validation)** est utilisée pour vérifier la stabilité des modèles et compléter l'évaluation sur le jeu de test.

Les expériences, les modèles et leurs métriques sont suivis avec **MLflow** :

http://194.238.26.226:5000

---

## GitHub Actions

**GitHub Actions** est utilisé pour automatiser les vérifications du projet à chaque modification du code.

Le pipeline CI/CD permet d'exécuter automatiquement les vérifications et de s'assurer que le projet continue de fonctionner avant d'intégrer les changements.

---

## Contraintes

Le projet présente certaines limites :

- les classes de dommages sont très déséquilibrées ;
- il existe peu de séismes classés Sévère ;
- toutes les données PAGER ne sont pas disponibles pour tous les séismes ;
- les pertes économiques sont disponibles pour un nombre limité d'événements ;
- les événements avec des pertes économiques très élevées sont rares ;
- les pertes économiques dépendent aussi de facteurs comme la population, les bâtiments et les infrastructures qui ne sont pas tous présents dans le dataset ;
- les modèles ont été développés et évalués avec les données de **2025**.

Les résultats doivent donc être considérés comme des estimations basées sur les données disponibles.

---

## Résumé

| Modèle | Objectif | Champion / Méthode | Résultat principal |
|---|---|---|---|
| Dommages | Prédire le niveau de dommages | Random Forest | Accuracy 98,9 %, F1 Sévère 0,80 |
| Action | Déterminer l'action | Règles de décision | 904 événements traités |
| Pertes économiques | Estimer les pertes économiques | Gradient Boosting | R² 0,6859 |

---

## Application Streamlit

Une interface Streamlit permet de tester le projet et de visualiser les résultats :

https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app/

---

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- USGS API
- USGS PAGER
- MLflow
- GitHub Actions
- Streamlit