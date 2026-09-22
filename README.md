# Risques Sismiques

Projet de Machine Learning réalisé à partir de données sismiques de l'USGS.

Le projet contient plusieurs modèles pour étudier les risques et les conséquences possibles des séismes.

## Application

Application Streamlit :

https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app

## Modèles

### Model 1 — Zone de risque

Classification des séismes en trois niveaux :

- Faible
- Modéré
- Sévère

### Model 3 — PGA

Prédiction du Peak Ground Acceleration (PGA).

### Model 4 — Facteurs de risque

Étude des variables liées au niveau de risque d'un séisme.

### Model 5 — Dommages

Le modèle Damage est un modèle de classification destiné à estimer le niveau de dommages d'un séisme.

La variable cible `Damage_Level` contient trois classes :

- Faible
- Modéré
- Sévère

Les classes sont construites à partir des informations USGS PAGER afin d'éviter de définir directement le niveau de dommages à partir de la magnitude.

#### Données utilisées

Le modèle travaille sur les événements pour lesquels une alerte PAGER est disponible.

Le dataset contient 904 événements PAGER :

- 865 événements — Faible
- 25 événements — Modéré
- 14 événements — Sévère

La distribution des classes est donc fortement déséquilibrée.

Les variables utilisées pour la classification sont :

- magnitude (`mag`)
- profondeur (`depth`)
- MMI (`mmi`)
- CDI (`cdi`)
- nombre de signalements (`felt`)
- SIG (`sig`)
- tsunami (`tsunami`)
- latitude (`latitude`)
- longitude (`longitude`)

#### Prétraitement

Les variables `CDI` et `FELT` contiennent des valeurs manquantes.

Les deux variables sont manquantes ensemble dans 451 observations.

Pour `FELT`, une transformation logarithmique `log1p()` est utilisée afin de réduire l'influence des valeurs très élevées.

Les valeurs manquantes sont ensuite remplacées par la médiane.

Après le prétraitement, aucune valeur manquante ne reste dans les variables utilisées par le modèle.

#### Séparation des données

Les données sont séparées en :

- 80 % pour l'entraînement
- 20 % pour le test

La séparation est stratifiée afin de conserver les trois classes dans les deux ensembles.

Résultat :

- 723 observations pour l'entraînement
- 181 observations pour le test

#### Algorithmes testés

Plusieurs algorithmes de classification sont étudiés :

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

KNN utilise des variables normalisées avec `StandardScaler`.

Les modèles basés sur les arbres utilisent les variables originales après prétraitement.

#### Résultat KNN

Accuracy :

**96,13 %**

Le modèle obtient une bonne accuracy globale, mais le Recall de la classe `Sévère` est de **33 %**.

Cela montre que l'accuracy seule ne suffit pas pour évaluer ce problème fortement déséquilibré.

#### Résultat Decision Tree

Accuracy :

**98,34 %**

Le Decision Tree utilise `class_weight='balanced'` afin de mieux prendre en compte les classes minoritaires.

Résultats sur le jeu de test :

- Faible : Recall 99 %
- Modéré : Recall 80 %
- Sévère : Recall 67 %

Le modèle détecte donc mieux les classes minoritaires que KNN.

#### Attention au déséquilibre des classes

Le dataset contient beaucoup plus d'événements `Faible` que d'événements `Modéré` ou `Sévère`.

La performance doit donc être analysée avec plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion

Le Recall de la classe `Sévère` est particulièrement important pour analyser la capacité du modèle à détecter les événements à impact élevé.

#### Objectif du modèle

L'objectif n'est pas simplement d'obtenir une accuracy élevée.

Le modèle doit être capable de distinguer les événements à faible impact des événements présentant un niveau de dommages plus important, tout en tenant compte du fort déséquilibre des classes.

Les résultats du notebook montrent l'importance de comparer plusieurs algorithmes plutôt que de se baser uniquement sur l'accuracy.
### Model 6 — Action

Détermination d'une action à partir du scénario sismique et des résultats des modèles.

### Model 7 — Pertes économiques

Prédiction des pertes économiques à partir des données USGS et PAGER.

## Version 1

Variables utilisées :

- magnitude
- profondeur
- SIG
- latitude
- longitude
- tsunami

Trois modèles ont été testés :

- Linear Regression
- Random Forest
- Gradient Boosting

Pour les séismes de magnitude M5.0 ou plus, **2 122 événements** ont été identifiés.

Les données économiques PAGER étaient disponibles pour **734 événements**.

Parmi ces 734 événements :

- **625 événements (85,15 %) avaient une perte économique nulle**
- **109 événements (14,85 %) avaient une perte économique positive**

Comme les pertes économiques sont très déséquilibrées, la variable cible a été transformée avec `log1p()` avant l'entraînement.

### Résultats

| Modèle | R² sur l'échelle logarithmique | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.5095 | $2.435 T | $29.519 T |
| Random Forest | **0.7092** | $358.05 M | $4.324 B |
| Gradient Boosting | 0.6859 | **$353.47 M** | **$4.266 B** |

Le Random Forest obtient le R² logarithmique le plus élevé avec **0.7092**.

Le Gradient Boosting obtient les plus faibles erreurs en dollars, avec une MAE de **$353.47 millions** et une RMSE de **$4.27 milliards**.

Pour les **22 événements du jeu de test ayant une perte économique positive**, le Gradient Boosting obtient une MAE de **$2.36 milliards** et une RMSE de **$11.03 milliards**.

Le **Gradient Boosting a été conservé comme modèle final**.

La variable la plus importante du modèle est `sig`, avec une importance d'environ **69.02 %**, suivie de la magnitude et de la position géographique.

### Limite importante

Le modèle reste une estimation expérimentale. Les pertes catastrophiques sont difficiles à prévoir avec seulement les caractéristiques d'un séisme.

Par exemple, pour le séisme du Myanmar dont la perte PAGER était d'environ **$52.99 milliards**, le modèle Gradient Boosting a prédit environ **$1.27 milliard**.

Pour améliorer le modèle, de futures versions pourraient utiliser des informations supplémentaires comme la population exposée, les infrastructures, les bâtiments, l'activité économique et la distance par rapport aux zones habitées.

## Version 2 — PAGER avec données d'exposition

La V2 ajoute les informations d'exposition provenant de PAGER.

Le dataset contient **734 événements** avec une estimation économique PAGER disponible.

Les variables utilisées sont :

- magnitude
- profondeur
- MMI maximale
- population exposée aux niveaux MMI 5 à 10
- exposition économique aux niveaux MMI 5 à 10

La V2 utilise deux modèles :

- un modèle de classification pour déterminer si une perte économique positive est prévue ;
- un modèle de régression pour estimer le montant de la perte.

### Validation croisée à 5 plis

#### Random Forest

- MAE : 3.0464
- RMSE : 3.7642
- R² : 0.6014

#### Gradient Boosting

- MAE : 3.0304
- RMSE : 3.8553
- R² : 0.5838

Les fichiers principaux de la V2 sont :

```text
data/economic_loss_model_v2_2025.csv
models/economic_loss_v2_classifier.pkl
models/economic_loss_v2_regressor.pkl
```

### Exemple

Pour l'événement USGS `us7000pn9s` :

- Magnitude : 7.7
- Profondeur : 10 km
- MMI maximale : 10
- Latitude : 22.0110
- Longitude : 95.9363

L'application a produit une estimation de :

**10,249,856,333.48 $ US**

Cette valeur est une estimation du modèle basée sur les données PAGER. Elle ne représente pas une perte économique finale confirmée.

## Expérimentation — Scénario futur

Une expérimentation a été réalisée pour essayer de prédire les pertes économiques d'un séisme hypothétique avec :

- latitude
- longitude
- magnitude
- profondeur

Le premier modèle a obtenu un R² d'environ **0.13**.

Avec la MMI observée, le R² était d'environ **0.68**.

La MMI prédite avait une corrélation de **0.8429** avec la MMI réelle.

Cependant, lorsque la MMI prédite a été utilisée pour prédire les pertes économiques :

- MAE : 5.0753
- RMSE : 5.8793
- R² : 0.0913

Cette approche n'est donc pas utilisée comme modèle principal de l'application.

## Données

Les données utilisées dans le projet proviennent principalement de l'USGS et de PAGER.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Folium
- Random Forest
- Gradient Boosting
- Git
- GitHub

## Installation

```text
git clone https://github.com/tarekalsemaan/USGS-Earthquake-AI.git
cd USGS-Earthquake-AI
pip install -r requirements.txt
```

## Lancer l'application

```text
streamlit run map/app.py
```

## Limites

Les pertes économiques utilisées dans le projet correspondent aux estimations PAGER disponibles.

Les résultats sont des estimations de Machine Learning et ne représentent pas nécessairement les pertes économiques finales réelles.

Les données PAGER ne sont pas disponibles pour tous les séismes.

Les modèles ont été réalisés dans le cadre de ce projet et ne remplacent pas les systèmes officiels d'évaluation des séismes et des dommages.

## GitHub

https://github.com/tarekalsemaan/USGS-Earthquake-AI
