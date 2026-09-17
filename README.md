Risques Sismiques

Projet de Machine Learning réalisé à partir de données sismiques de l'USGS.

Le projet contient plusieurs modèles pour étudier les risques et les conséquences possibles des séismes.

Application

Application Streamlit :

https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app

Modèles
Model 1 — Zone de risque

Classification des séismes en trois niveaux :

Faible
Modéré
Sévère
Model 3 — PGA

Prédiction du Peak Ground Acceleration (PGA).

Model 4 — Facteurs de risque

Étude des variables liées au niveau de risque d'un séisme.

Model 5 — Dommages

Prédiction du niveau de dommages :

Faible
Modéré
Sévère

Les principales variables utilisées sont la magnitude, la profondeur, la MMI, le CDI, le nombre de signalements, le SIG, le tsunami et la position géographique.

Model 6 — Action

Détermination d'une action à partir du scénario sismique et des résultats des modèles.

Model 7 — Pertes économiques

Prédiction des pertes économiques à partir des données USGS et PAGER.

Version 1

Variables utilisées :

magnitude ;
profondeur ;
SIG ;
latitude ;
longitude ;
tsunami.

Trois modèles ont été testés :

Linear Regression ;
Random Forest ;
Gradient Boosting.

Résultats :

Modèle	MAE	RMSE	R²
Linear Regression	1.6604	3.0105	0.5117
Random Forest	1.0040	2.4327	0.6812
Gradient Boosting	1.0387	2.2755	0.7210
Version 2 — PAGER

La V2 ajoute les informations d'exposition provenant de PAGER.

Le dataset contient 734 événements avec une estimation économique PAGER disponible.

Les variables utilisées sont :

magnitude ;
profondeur ;
MMI maximale ;
population exposée aux niveaux MMI 5 à 10 ;
exposition économique aux niveaux MMI 5 à 10.

La V2 utilise deux modèles :

un modèle de classification pour déterminer si une perte économique positive est prévue ;
un modèle de régression pour estimer le montant de la perte.

Validation croisée à 5 plis :

Random Forest

MAE : 3.0464
RMSE : 3.7642
R² : 0.6014

Gradient Boosting

MAE : 3.0304
RMSE : 3.8553
R² : 0.5838

Les fichiers principaux de la V2 sont :

data/economic_loss_model_v2_2025.csv

models/economic_loss_v2_classifier.pkl

models/economic_loss_v2_regressor.pkl

Exemple

Pour l'événement USGS us7000pn9s :

Magnitude : 7.7
Profondeur : 10 km
MMI maximale : 10
Latitude : 22.0110
Longitude : 95.9363

L'application a produit une estimation de :

10,249,856,333.48 $ US

Cette valeur est une estimation du modèle basée sur les données PAGER. Elle ne représente pas une perte économique finale confirmée.

Expérimentation — Scénario futur

Une expérimentation a été réalisée pour essayer de prédire les pertes économiques d'un séisme hypothétique avec :

latitude ;
longitude ;
magnitude ;
profondeur.

Le premier modèle a obtenu un R² d'environ 0.13.

Avec la MMI observée, le R² était d'environ 0.68.

La MMI prédite avait une corrélation de 0.8429 avec la MMI réelle.

Cependant, lorsque la MMI prédite a été utilisée pour prédire les pertes économiques :

MAE : 5.0753
RMSE : 5.8793
R² : 0.0913

Cette approche n'est donc pas utilisée comme modèle principal de l'application.

Données

Les données utilisées dans le projet proviennent principalement de l'USGS et de PAGER.

Technologies
Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Streamlit
Folium
Random Forest
Gradient Boosting
Git
GitHub
Installation
git clone https://github.com/tarekalsemaan/USGS-Earthquake-AI.git
cd USGS-Earthquake-AI
pip install -r requirements.txt
Lancer l'application
streamlit run map/app.py
Limites

Les pertes économiques utilisées dans le projet correspondent aux estimations PAGER disponibles.

Les résultats sont des estimations de Machine Learning et ne représentent pas nécessairement les pertes économiques finales réelles.

Les données PAGER ne sont pas disponibles pour tous les séismes.

Les modèles ont été réalisés dans le cadre de ce projet et ne remplacent pas les systèmes officiels d'évaluation des séismes et des dommages.

GitHub

https://github.com/tarekalsemaan/USGS-Earthquake-AI