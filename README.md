# USGS Earthquake AI

Projet de Machine Learning réalisé à partir des données sismiques de l'USGS et de PAGER.

L'objectif du projet est d'analyser les séismes sous plusieurs angles : préparation des données, zone de risque, PGA, facteurs de risque, dommages, actions recommandées et pertes économiques.

## Application Streamlit

L'application permet de tester les modèles à partir d'un séisme et d'afficher les résultats de façon simple sur une carte.

https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app/

---

## Notebooks

### 01 — EDA et prétraitement

`01_EDA_Preprocessing.ipynb`

Ce notebook prépare les données USGS avant l'entraînement des modèles.

Il sert principalement à explorer les données, vérifier les valeurs manquantes, nettoyer les variables et préparer les caractéristiques qui seront utilisées dans les notebooks suivants.

---

### 02 — Zone de risque

`02_Model_Risk_Zone.ipynb`

Ce notebook cherche à classer les séismes selon leur niveau de risque :

- Faible
- Modéré
- Sévère

Plusieurs approches de classification sont étudiées. Un modèle Random Forest est notamment sauvegardé dans :

`models/Risk_Zone_RandomForest_champion_model.joblib`

---

### 03 — PGA

`03_Model_PGA.ipynb`

Ce notebook travaille sur le **Peak Ground Acceleration (PGA)**.

Le PGA représente l'accélération maximale du sol pendant un séisme. Le but est d'estimer cette valeur à partir des caractéristiques disponibles de l'événement sismique.

---

### 04 — Facteurs de risque

`04_Model_Risk_Factors.ipynb`

Ce notebook analyse les facteurs qui peuvent être associés au risque sismique.

Les variables étudiées comprennent notamment la magnitude, la profondeur, la position géographique, le SIG, le tsunami et la MMI.

L'objectif est surtout de comprendre quelles caractéristiques ont le plus d'influence dans l'analyse du risque.

---

### 05 — Dommages

`05_Model_Damage.ipynb`

Ce notebook prédit le niveau de dommages d'un séisme.

Trois classes sont utilisées :

- Faible
- Modéré
- Sévère

Le dataset contient **904 événements PAGER** :

| Niveau | Nombre |
| --- | ---: |
| Faible | 865 |
| Modéré | 25 |
| Sévère | 14 |

Les modèles testés comprennent KNN, Decision Tree et Random Forest.

**Résultats principaux :**

| Modèle | Accuracy |
| --- | ---: |
| KNN | 96,13 % |
| Decision Tree | 98,34 % |

Pour le Decision Tree, le recall obtenu est de **99 % pour Faible, 80 % pour Modéré et 67 % pour Sévère**.

Ces résultats montrent aussi pourquoi l'accuracy seule n'est pas suffisante : les classes Modéré et Sévère sont beaucoup moins nombreuses.

---

### 06 — Action

`06_Model_Action.ipynb`

Ce notebook transforme le niveau de dommages en une recommandation opérationnelle.

Contrairement aux autres modèles, il ne s'agit pas d'un modèle de Machine Learning entraîné. C'est un moteur de décision basé sur des règles simples et interprétables.

| Dommages | Action | Priorité |
| --- | --- | --- |
| Faible | Surveillance | Faible |
| Modéré | Évaluation prioritaire | Moyenne |
| Sévère | Intervention prioritaire | Haute |

Le tsunami est utilisé comme avertissement supplémentaire.

**Résultats sur les 904 événements :**

| Action | Nombre |
| --- | ---: |
| Surveillance | 865 |
| Évaluation prioritaire | 25 |
| Intervention prioritaire | 14 |

**107 événements** ont également généré un avertissement tsunami.

Le moteur est disponible dans `models/action_engine.py`.

---

### 07 — Pertes économiques

`07_Model_Economic_Loss.ipynb`

Ce notebook estime les pertes économiques associées aux séismes à partir des données USGS et PAGER.

Pour les séismes de magnitude 5 ou plus, **2 122 événements** ont été identifiés. Les données économiques PAGER étaient disponibles pour **734 événements**.

Parmi ces 734 événements :

- 625 avaient une perte économique nulle ;
- 109 avaient une perte économique positive.

Trois modèles ont été comparés :

| Modèle | R² log | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.5095 | $2.435 T | $29.519 T |
| Random Forest | 0.7092 | $358.05 M | $4.324 B |
| Gradient Boosting | 0.6859 | $353.47 M | $4.266 B |

Le **Gradient Boosting** est utilisé comme modèle final de la Version 1 car il obtient les erreurs en dollars les plus faibles.

La variable la plus importante est `sig`, avec environ **69 % d'importance**.

Une Version 2 a également été développée avec des variables d'exposition PAGER et deux étapes : déterminer s'il existe une perte positive, puis estimer son montant.

---

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Folium
- Git
- GitHub

## Lancer le projet

```bash
pip install -r requirements.txt
streamlit run map/app.py
```

## Limites

Les résultats sont des estimations produites à partir des données disponibles et de modèles expérimentaux.

Les données PAGER ne sont pas disponibles pour tous les séismes. Les pertes économiques réelles peuvent également être très différentes des estimations du modèle.

Le moteur d'action du Model 6 a été créé pour ce projet et ne représente pas un protocole officiel d'intervention d'urgence.
