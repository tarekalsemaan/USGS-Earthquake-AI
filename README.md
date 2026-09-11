# USGS-Earthquake-AI

Projet de Machine Learning réalisé à partir des données sismiques de l’USGS.

Le projet permet d’analyser les séismes et de faire des prédictions concernant :

* les zones de risque ;
* le PGA ;
* les dommages ;
* les actions à prendre ;
* les pertes économiques.

## Application

L’application est disponible en ligne avec Streamlit :

https://usgs-earthquake-ai-9zbupyrcn6cwtmhms5xa9p.streamlit.app

---

## Model 7 — Economic Loss Prediction

This model estimates the economic losses associated with significant earthquakes.

### Data

Earthquake characteristics are obtained from the **USGS 2025 earthquake dataset**.

Economic-loss targets are obtained from **USGS PAGER (Prompt Assessment of Global Earthquakes for Response)**.

PAGER provides estimated economic impacts for significant earthquakes.

After matching the available PAGER information with the USGS earthquake events:

- Significant earthquakes analyzed: 2,122
- Events with available PAGER economic-loss estimates: 734
- Events without economic-loss estimates: 1,388

The final Machine Learning dataset therefore contains **734 earthquakes**.

### Variables

The model uses the following explanatory variables:

- `mag` — earthquake magnitude
- `depth` — earthquake depth
- `sig` — USGS significance index
- `latitude`
- `longitude`
- `tsunami`

Target variable:

- `economic_loss` — estimated economic loss from USGS PAGER

### Preprocessing

Economic losses are highly skewed, ranging from $0 to several billion dollars.

A logarithmic transformation was therefore applied:

`log1p(economic_loss)`

The dataset was divided into:

- 80% training data
- 20% test data

### Algorithms

Three regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.6604 | 3.0105 | 0.5117 |
| Random Forest | 1.0040 | 2.4327 | 0.6812 |
| **Gradient Boosting** | **1.0387** | **2.2755** | **0.7210** |

The metrics above are calculated on the logarithmically transformed target.

**Gradient Boosting** achieved the best overall performance, with an R² of **0.7210** and the lowest RMSE.

### Feature Importance

The Gradient Boosting model identified the following feature importances:

| Variable | Importance |
|---|---:|
| `sig` | 68.84% |
| `mag` | 12.12% |
| `longitude` | 10.44% |
| `latitude` | 5.41% |
| `depth` | 2.88% |
| `tsunami` | 0.31% |

### Saved Model

The trained model is available at:

`models/economic_loss_gradient_boosting.pkl`

The final modeling dataset is available at:

`data/economic_loss_model_2025.csv`

The downloaded PAGER economic-loss dataset is available at:

`data/pager_economic_losses_2025.csv`

### Important Limitation

The model predicts **USGS PAGER economic-loss estimates**, not confirmed final economic losses.

The results should therefore be interpreted as an experimental Machine Learning approximation of PAGER estimates rather than an exact prediction of the financial damage caused by a future earthquake.

