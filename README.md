# Risques Sismiques — USGS Earthquake AI



Projet de Machine Learning réalisé à partir de données sismiques provenant principalement de l'USGS et de PAGER.



L'objectif du projet est d'analyser différents aspects du risque sismique :



- zone de risque ;

- PGA ;

- facteurs de risque ;

- dommages ;

- action opérationnelle ;

- pertes économiques.



Le projet combine Machine Learning, analyse de données, visualisation et moteur de décision.



---



## Application



Application Streamlit :



https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app/



---



# Architecture du projet



Le projet est organisé en plusieurs étapes :



```text

Données USGS

     ↓

Prétraitement

     ↓

Analyse sismique

     ↓

Modèles de Machine Learning

     ↓

Dommages

     ↓

Moteur de décision

     ↓

Pertes économiques

     ↓

Application Streamlit

```



Les notebooks principaux sont :



```text

notebooks/

│

├── 01_EDA_Preprocessing

├── 02_Model_Risk_Zone

├── 03_Model_PGA

├── 04_Model_Risk_Factors

├── 05_Model_Damage

├── 06_Model_Action

└── 07_Model_Economic_Loss

```



---



# Modèles



## Model 1 — Zone de risque



Classification des séismes en trois niveaux :



- Faible

- Modéré

- Sévère



Le projet utilise notamment un Random Forest pour la classification de la zone de risque.



Le modèle sauvegardé est disponible dans :



```text

models/Risk_Zone_RandomForest_champion_model.joblib

```



---



## Model 3 — PGA



Le Model 3 porte sur la prédiction du :



****Peak Ground Acceleration (PGA)****



Le PGA représente l'accélération maximale du sol associée à un événement sismique.



L'objectif est d'utiliser les caractéristiques disponibles d'un séisme afin d'estimer l'intensité du mouvement du sol.



---



## Model 4 — Facteurs de risque



Le Model 4 étudie les variables associées au niveau de risque d'un séisme.



L'objectif est d'identifier les caractéristiques sismiques importantes et d'analyser leur relation avec le risque.



Les variables étudiées comprennent notamment :



- magnitude ;

- profondeur ;

- latitude ;

- longitude ;

- SIG ;

- tsunami ;

- MMI ;

- autres caractéristiques USGS disponibles.



---



# Model 5 — Dommages



Le Model 5 est un modèle de classification destiné à estimer le niveau de dommages d'un séisme.



La variable cible `Damage_Level` contient trois classes :



- Faible

- Modéré

- Sévère



Les classes sont construites à partir des informations USGS PAGER afin d'éviter de définir directement le niveau de dommages uniquement à partir de la magnitude.



---



## Données utilisées



Le dataset contient 904 événements pour lesquels les informations PAGER nécessaires sont disponibles.



Distribution :



| Damage_Level | Nombre |

|---|---:|

| Faible | 865 |

| Modéré | 25 |

| Sévère | 14 |

| ****Total**** | ****904**** |



La distribution des classes est fortement déséquilibrée.



---



## Variables utilisées



Les principales variables utilisées pour la classification sont :



- magnitude (`mag`)

- profondeur (`depth`)

- MMI (`mmi`)

- CDI (`cdi`)

- nombre de signalements (`felt`)

- SIG (`sig`)

- tsunami (`tsunami`)

- latitude (`latitude`)

- longitude (`longitude`)



---



## Prétraitement



Les variables `CDI` et `FELT` contiennent des valeurs manquantes.



Les deux variables sont manquantes ensemble dans 451 observations.



Pour `FELT`, une transformation logarithmique `log1p()` est utilisée afin de réduire l'influence des valeurs très élevées.



Les valeurs manquantes sont ensuite remplacées par la médiane.



Des variables prétraitées sont également utilisées :



```text

felt_log

cdi_imputed

felt_log_imputed

```



Après le prétraitement, aucune valeur manquante ne reste dans les variables utilisées par le modèle de classification.



---



## Séparation des données



Les données sont séparées en :



- 80 % pour l'entraînement ;

- 20 % pour le test.



La séparation est stratifiée afin de conserver les classes dans les ensembles d'entraînement et de test.



Résultat :



- 723 observations pour l'entraînement ;

- 181 observations pour le test.



---



## Algorithmes testés



Plusieurs algorithmes de classification sont étudiés :



- K-Nearest Neighbors (KNN)

- Decision Tree

- Random Forest



KNN utilise des variables normalisées avec `StandardScaler`.



Les modèles basés sur les arbres utilisent les variables après prétraitement.



---



## Résultat KNN



Accuracy :



****96,13 %****



Le modèle obtient une bonne accuracy globale.



Cependant, le Recall de la classe `Sévère` est de ****33 %****.



Cela montre que l'accuracy seule ne suffit pas pour évaluer correctement un problème fortement déséquilibré.



---



## Résultat Decision Tree



Accuracy :



****98,34 %****



Le Decision Tree utilise :



```python

class_weight="balanced"

```



afin de mieux prendre en compte les classes minoritaires.



Résultats sur le jeu de test :



| Classe | Recall |

|---|---:|

| Faible | 99 % |

| Modéré | 80 % |

| Sévère | 67 % |



---



## Évaluation



En raison du déséquilibre des classes, plusieurs métriques sont utilisées :



- Accuracy

- Precision

- Recall

- F1-score

- Matrice de confusion



Le Recall de la classe `Sévère` est particulièrement important pour analyser la capacité du système à détecter les événements présentant un niveau de dommages élevé.



Le modèle sauvegardé est disponible dans :



```text

models/damage_random_forest.joblib

models/damage_scaler.joblib

```



---



# Model 6 — Action



Le Model 6 constitue la couche de décision opérationnelle du projet.



Contrairement aux modèles prédictifs précédents, il ne s'agit pas d'un modèle de Machine Learning entraîné sur une cible `Action`.



Le Model 6 utilise un ****moteur de décision basé sur des règles explicites****.



Son rôle est de transformer le niveau de dommages estimé en recommandation opérationnelle.



---



## Fonctionnement



Le moteur reçoit :



- `Damage_Level`

- `tsunami`



et produit :



- l'action recommandée ;

- le niveau de priorité ;

- un avertissement tsunami éventuel ;

- une justification.



---



## Règles de décision



| Niveau de dommages | Action | Priorité |

|---|---|---|

| Faible | Surveillance | Faible |

| Modéré | Évaluation prioritaire | Moyenne |

| Sévère | Intervention prioritaire | Haute |



L'indicateur `tsunami` est utilisé comme avertissement complémentaire.



Il ne provoque pas automatiquement une modification du niveau d'action.



---



## Module



Le moteur est disponible dans :



```text

models/action_engine.py

```



La sortie complète est sauvegardée dans :



```text

data/action_model_2025.csv

```



---



## Validation



Le moteur a été appliqué aux 904 événements du dataset de dommages.



Résultats :



| Action | Nombre |

|---|---:|

| Surveillance | 865 |

| Évaluation prioritaire | 25 |

| Intervention prioritaire | 14 |

| ****Total**** | ****904**** |



Avertissements tsunami :



| État | Nombre |

|---|---:|

| Aucun avertissement tsunami | 797 |

| Avertissement tsunami | 107 |

| ****Total**** | ****904**** |



Les contrôles automatiques ont confirmé :



- 904 événements traités ;

- aucune action manquante ;

- aucune priorité manquante ;

- aucune action indéterminée ;

- aucun doublon `event_id` ;

- aucune incohérence dans l'application des règles.



Le Model 6 est donc un ****moteur de décision déterministe et interprétable****, et non un modèle ML supplémentaire.



---



# Model 7 — Pertes économiques



Le Model 7 porte sur la prédiction des pertes économiques associées aux séismes.



Deux versions du problème ont été étudiées.



---



## Version 1 — Pertes économiques



Les variables utilisées sont :



- magnitude (`mag`)

- profondeur (`depth`)

- SIG (`sig`)

- latitude (`latitude`)

- longitude (`longitude`)

- tsunami (`tsunami`)



La cible économique est transformée avec `log1p()` avant l'entraînement afin de réduire l'influence des pertes extrêmement élevées.



Les prédictions sont ensuite reconverties sur l'échelle originale en dollars américains.



---



## Données



Pour les séismes de magnitude M5.0 ou plus :



****2 122 événements**** ont été identifiés.



Les données économiques PAGER étaient disponibles pour :



****734 événements****



Parmi ces 734 événements :



- 625 événements (85,15 %) avaient une perte économique nulle ;

- 109 événements (14,85 %) avaient une perte économique positive.



Cette concentration importante de pertes nulles constitue une limite du problème.



---



## Modèles testés



Trois modèles ont été comparés :



- Linear Regression

- Random Forest

- Gradient Boosting



---



## Résultats



| Modèle | R² logarithmique | MAE | RMSE |

|---|---:|---:|---:|

| Linear Regression | 0.5095 | $2.435 T | $29.519 T |

| Random Forest | 0.7092 | $358.05 M | $4.324 B |

| Gradient Boosting | 0.6859 | $353.47 M | $4.266 B |



Le Random Forest obtient le R² logarithmique le plus élevé :



****0.7092****



Le Gradient Boosting obtient les plus faibles erreurs exprimées en dollars :



- MAE : ****$353.47 millions****

- RMSE : ****$4.27 milliards****



Pour les 22 événements du jeu de test ayant une perte économique positive, le Gradient Boosting obtient :



- MAE : ****$2.36 milliards****

- RMSE : ****$11.03 milliards****



Le Gradient Boosting est utilisé comme modèle final de la Version 1.



Le modèle est disponible dans :



```text

models/economic_loss_gradient_boosting.pkl

```



---



## Importance des variables



La variable la plus importante du modèle est `sig`, avec une importance d'environ 69,02 %.



Elle est suivie notamment par :



- magnitude ;

- longitude ;

- latitude ;

- profondeur ;

- tsunami.



---



## Exemple



Pour l'événement USGS :



```text

us7000pn9s

```



Caractéristiques :



- Magnitude : 7.7

- Profondeur : 10 km

- SIG : 2910

- Latitude : 22.0110

- Longitude : 95.9363

- Tsunami : Non



L'application produit une estimation d'environ :



****$1.27 milliard US****



Cette valeur est une estimation produite par le modèle Gradient Boosting.



Elle ne représente pas une perte économique finale confirmée.



---



# Version 2 — Pertes économiques avec exposition PAGER



Une deuxième version du modèle intègre des variables d'exposition provenant de PAGER.



Le dataset contient notamment :



- `maxmmi`

- `population_mmi_5`

- `population_mmi_6`

- `population_mmi_7`

- `population_mmi_8`

- `population_mmi_9`

- `population_mmi_10`

- `economic_exposure_mmi_5`

- `economic_exposure_mmi_6`

- `economic_exposure_mmi_7`

- `economic_exposure_mmi_8`

- `economic_exposure_mmi_9`

- `economic_exposure_mmi_10`



La Version 2 distingue deux problèmes :



1\. déterminer si une perte économique positive est présente ;

2\. estimer le montant de la perte économique.



Les modèles sont sauvegardés dans :



```text

models/economic_loss_v2_classifier.pkl

models/economic_loss_v2_regressor.pkl

```



Les données sont disponibles dans :



```text

data/economic_loss_model_v2_2025.csv

```



---



# Données



Les données utilisées dans le projet proviennent principalement de :



- USGS ;

- USGS PAGER ;

- produits sismiques associés lorsque disponibles.



Les données PAGER ne sont pas disponibles pour tous les événements.



Les principaux fichiers de données sont :



```text

data/

│

├── usgs_earthquakes_2025.csv

├── usgs_damage_2025.csv

├── usgs_damage_processed_2025.csv

├── pager_economic_losses_2025.csv

├── economic_loss_model_2025.csv

├── economic_loss_model_v2_2025.csv

└── action_model_2025.csv

```



---



# Technologies



- Python

- Pandas

- NumPy

- Scikit-learn

- Jupyter Notebook

- Streamlit

- Folium

- Random Forest

- Gradient Boosting

- K-Nearest Neighbors

- Decision Tree

- Git

- GitHub



---



# Installation



Cloner le dépôt :



```text

git clone https://github.com/tarekalsemaan/USGS-Earthquake-AI.git

cd USGS-Earthquake-AI

```



Installer les dépendances :



```text

pip install -r requirements.txt

```



---



# Lancer l'application



Depuis la racine du projet :



```text

streamlit run map/app.py

```



---



# Structure du projet



```text

USGS-Earthquake-AI/

│

├── .github/

├── data/

├── map/

├── models/

├── notebooks/

├── README.md

├── requirements.txt

└── tp_27-08-2026.docx

```



---



# Limites du projet



Les résultats du projet sont des estimations produites à partir de données historiques et de modèles de Machine Learning.



Les pertes économiques utilisées correspondent aux estimations PAGER disponibles et ne représentent pas nécessairement les pertes économiques finales réelles.



Les données PAGER ne sont pas disponibles pour tous les séismes.



Les pertes économiques présentent une distribution fortement déséquilibrée, avec un grand nombre d'événements sans perte ou avec une perte nulle et un nombre limité d'événements présentant des pertes extrêmement élevées.



Les événements extrêmes restent particulièrement difficiles à prédire.



Le Model 6 utilise un moteur de décision basé sur des règles définies dans le cadre de ce projet. Il ne constitue pas un protocole officiel d'intervention d'urgence.



Les modèles développés dans ce projet sont expérimentaux et ne remplacent pas les systèmes officiels d'évaluation des séismes, des dommages ou des interventions d'urgence.



---



# GitHub



Dépôt du projet :



https://github.com/tarekalsemaan/USGS-Earthquake-AI
