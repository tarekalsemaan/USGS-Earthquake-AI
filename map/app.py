import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import requests

from datetime import date
from streamlit_folium import st_folium


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="USGS Earthquake AI",
    layout="wide"
)


# ============================================================
# MODELS
# ============================================================

damage_model = joblib.load(
    "models/damage_random_forest.joblib"
)

economic_loss_classifier = joblib.load(
    "models/economic_loss_v2_classifier.pkl"
)

economic_loss_regressor = joblib.load(
    "models/economic_loss_v2_regressor.pkl"
)


# ============================================================
# USGS PAGER
# ============================================================

def load_usgs_pager_data(event_id):

    event_id = event_id.strip()

    event_url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?eventid={event_id}&format=geojson"
    )

    response = requests.get(
        event_url,
        timeout=15
    )

    response.raise_for_status()

    event_data = response.json()

    properties = event_data["properties"]
    geometry = event_data["geometry"]

    mag = properties["mag"]

    longitude = geometry["coordinates"][0]
    latitude = geometry["coordinates"][1]
    depth = geometry["coordinates"][2]

    products = properties.get(
        "products",
        {}
    )

    if "losspager" not in products:

        raise ValueError(
            "Aucune donnée PAGER disponible pour ce séisme."
        )

    pager_product = products["losspager"][0]

    contents = pager_product["contents"]

    if "json/exposures.json" not in contents:

        raise ValueError(
            "Les données d'exposition PAGER ne sont pas disponibles."
        )

    if "json/event.json" not in contents:

        raise ValueError(
            "Les données PAGER de l'événement ne sont pas disponibles."
        )

    exposures_url = contents[
        "json/exposures.json"
    ]["url"]

    pager_event_url = contents[
        "json/event.json"
    ]["url"]

    exposure_response = requests.get(
        exposures_url,
        timeout=15
    )

    exposure_response.raise_for_status()

    exposures = exposure_response.json()

    pager_response = requests.get(
        pager_event_url,
        timeout=15
    )

    pager_response.raise_for_status()

    pager_event = pager_response.json()

    maxmmi = float(
        pager_event["pager"]["maxmmi"]
    )

    population = exposures[
        "population_exposure"
    ]

    economic = exposures[
        "economic_exposure"
    ]

    population_exposure = population[
        "aggregated_exposure"
    ]

    economic_exposure = economic[
        "aggregated_exposure"
    ]

    result = {
        "event_id": event_id,
        "mag": float(mag),
        "depth": float(depth),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "maxmmi": maxmmi
    }

    for mmi_level in range(5, 11):

        index = mmi_level - 1

        result[
            f"population_mmi_{mmi_level}"
        ] = float(
            population_exposure[index]
        )

        result[
            f"economic_exposure_mmi_{mmi_level}"
        ] = float(
            economic_exposure[index]
        )

    return result


# ============================================================
# SESSION
# ============================================================

if "latitude" not in st.session_state:
    st.session_state.latitude = 45.5017

if "longitude" not in st.session_state:
    st.session_state.longitude = -73.5673

if "damage_prediction" not in st.session_state:
    st.session_state.damage_prediction = None

if "economic_prediction" not in st.session_state:
    st.session_state.economic_prediction = None

if "pager_data" not in st.session_state:
    st.session_state.pager_data = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "USGS Earthquake AI"
)

model_choice = st.sidebar.radio(
    "Choisir un modèle",
    [
        "Prédiction des dommages",
        "Prédiction des pertes économiques"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Projet de Machine Learning basé sur "
    "les données sismiques de l'USGS."
)


# ============================================================
# MODEL 1 — DAMAGE
# ============================================================

if model_choice == "Prédiction des dommages":

    st.title(
        "USGS Earthquake AI — Prédiction des dommages"
    )

    st.write(
        "Sélectionnez un emplacement sur la carte, "
        "entrez les caractéristiques du séisme, "
        "puis lancez la prédiction."
    )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    st.subheader(
        "1. Choisir une date"
    )

    selected_date = st.date_input(
        "Date",
        value=date.today()
    )

    st.caption(
        "La date identifie le scénario. "
        "Elle n'est pas utilisée directement par le modèle."
    )


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.subheader(
        "2. Choisir un emplacement"
    )

    st.write(
        "Cliquez sur la carte pour sélectionner "
        "l'emplacement du séisme."
    )

    selection_map = folium.Map(
        location=[
            st.session_state.latitude,
            st.session_state.longitude
        ],
        zoom_start=3
    )

    folium.CircleMarker(
        location=[
            st.session_state.latitude,
            st.session_state.longitude
        ],
        radius=8,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.8,
        tooltip="Emplacement sélectionné"
    ).add_to(selection_map)

    map_data = st_folium(
        selection_map,
        width=None,
        height=500,
        key="damage_selection_map",
        returned_objects=[
            "last_clicked"
        ]
    )

    if map_data and map_data.get(
        "last_clicked"
    ):

        clicked_latitude = map_data[
            "last_clicked"
        ]["lat"]

        clicked_longitude = map_data[
            "last_clicked"
        ]["lng"]

        location_changed = (
            abs(
                clicked_latitude
                - st.session_state.latitude
            ) > 0.0001
            or
            abs(
                clicked_longitude
                - st.session_state.longitude
            ) > 0.0001
        )

        if location_changed:

            st.session_state.latitude = (
                clicked_latitude
            )

            st.session_state.longitude = (
                clicked_longitude
            )

            st.session_state.damage_prediction = None

            st.rerun()


    col_lat, col_lon = st.columns(2)

    col_lat.metric(
        "Latitude",
        round(
            st.session_state.latitude,
            4
        )
    )

    col_lon.metric(
        "Longitude",
        round(
            st.session_state.longitude,
            4
        )
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.subheader(
        "3. Caractéristiques du séisme"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        mag = st.number_input(
            "Magnitude",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        depth = st.number_input(
            "Profondeur (km)",
            min_value=0.0,
            max_value=800.0,
            value=10.0,
            step=1.0
        )

        mmi = st.number_input(
            "MMI",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.1
        )

    with col2:

        cdi = st.number_input(
            "CDI — Intensité ressentie",
            min_value=0.0,
            max_value=10.0,
            value=4.1,
            step=0.1
        )

        felt = st.number_input(
            "Nombre de signalements ressentis",
            min_value=0,
            value=10,
            step=1
        )

        sig = st.number_input(
            "Indice de signification USGS (SIG)",
            min_value=0,
            value=400,
            step=10
        )

    with col3:

        tsunami_option = st.selectbox(
            "Tsunami",
            [
                "Non",
                "Oui"
            ]
        )

        tsunami = (
            1
            if tsunami_option == "Oui"
            else 0
        )

        st.write(
            "Emplacement"
        )

        st.write(
            f"Latitude : "
            f"{st.session_state.latitude:.4f}"
        )

        st.write(
            f"Longitude : "
            f"{st.session_state.longitude:.4f}"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "4. Prédiction"
    )

    if st.button(
        "Prédire les dommages",
        type="primary"
    ):

        felt_log_imputed = np.log1p(
            felt
        )

        damage_features = pd.DataFrame(
            [
                {
                    "mag": mag,
                    "depth": depth,
                    "mmi": mmi,
                    "cdi_imputed": cdi,
                    "felt_log_imputed": felt_log_imputed,
                    "sig": sig,
                    "tsunami": tsunami,
                    "latitude": st.session_state.latitude,
                    "longitude": st.session_state.longitude
                }
            ]
        )

        prediction = damage_model.predict(
            damage_features
        )[0]

        st.session_state.damage_prediction = (
            prediction
        )


    # --------------------------------------------------------
    # DAMAGE RESULT
    # --------------------------------------------------------

    if (
        st.session_state.damage_prediction
        is not None
    ):

        prediction = (
            st.session_state.damage_prediction
        )

        st.write(
            f"### Résultat pour le {selected_date}"
        )

        if prediction == "Faible":

            st.success(
                "Dommages prédits : FAIBLES"
            )

            marker_color = "green"

        elif prediction == "Modéré":

            st.warning(
                "Dommages prédits : MODÉRÉS"
            )

            marker_color = "orange"

        else:

            st.error(
                "Dommages prédits : SÉVÈRES"
            )

            marker_color = "red"

        result_map = folium.Map(
            location=[
                st.session_state.latitude,
                st.session_state.longitude
            ],
            zoom_start=6
        )

        folium.CircleMarker(
            location=[
                st.session_state.latitude,
                st.session_state.longitude
            ],
            radius=15,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8,
            tooltip=(
                f"Dommages prédits : "
                f"{prediction}"
            ),
            popup=folium.Popup(
                f"""
                <b>Date :</b> {selected_date}<br>
                <b>Magnitude :</b> {mag}<br>
                <b>Profondeur :</b> {depth} km<br>
                <b>MMI :</b> {mmi}<br>
                <b>CDI :</b> {cdi}<br>
                <b>SIG :</b> {sig}<br>
                <b>Dommages :</b> {prediction}
                """,
                max_width=350
            )
        ).add_to(result_map)

        st_folium(
            result_map,
            width=None,
            height=500,
            key="damage_result_map",
            returned_objects=[]
        )


# ============================================================
# MODEL 2 — ECONOMIC LOSS V2
# ============================================================

else:

    st.title(
        "USGS Earthquake AI — "
        "Prédiction des pertes économiques"
    )

    st.write(
        "Entrez l'identifiant d'un séisme USGS. "
        "L'application récupère automatiquement "
        "les données PAGER et ShakeMap nécessaires "
        "au modèle."
    )

    st.info(
        "Le modèle V2 estime les pertes à partir "
        "des données USGS PAGER / ShakeMap. "
        "Il s'agit d'une estimation de Machine Learning "
        "et non d'une perte économique réelle confirmée."
    )


    # --------------------------------------------------------
    # EVENT ID
    # --------------------------------------------------------

    st.subheader(
        "1. Séisme USGS"
    )

    event_id = st.text_input(
        "USGS Event ID",
        value="us7000pn9s",
        placeholder="Exemple : us7000pn9s"
    )

    st.caption(
        "Exemple de test : us7000pn9s"
    )


    # --------------------------------------------------------
    # LOAD PAGER + PREDICT
    # --------------------------------------------------------

    if st.button(
        "Analyser le séisme",
        type="primary"
    ):

        try:

            with st.spinner(
                "Chargement des données USGS PAGER..."
            ):

                pager_data = load_usgs_pager_data(
                    event_id
                )

            st.session_state.pager_data = pager_data

            economic_features = pd.DataFrame(
                [
                    {
                        "mag":
                            pager_data["mag"],

                        "depth":
                            pager_data["depth"],

                        "maxmmi":
                            pager_data["maxmmi"],

                        "population_mmi_5":
                            pager_data["population_mmi_5"],

                        "population_mmi_6":
                            pager_data["population_mmi_6"],

                        "population_mmi_7":
                            pager_data["population_mmi_7"],

                        "population_mmi_8":
                            pager_data["population_mmi_8"],

                        "population_mmi_9":
                            pager_data["population_mmi_9"],

                        "population_mmi_10":
                            pager_data["population_mmi_10"],

                        "economic_exposure_mmi_5":
                            pager_data[
                                "economic_exposure_mmi_5"
                            ],

                        "economic_exposure_mmi_6":
                            pager_data[
                                "economic_exposure_mmi_6"
                            ],

                        "economic_exposure_mmi_7":
                            pager_data[
                                "economic_exposure_mmi_7"
                            ],

                        "economic_exposure_mmi_8":
                            pager_data[
                                "economic_exposure_mmi_8"
                            ],

                        "economic_exposure_mmi_9":
                            pager_data[
                                "economic_exposure_mmi_9"
                            ],

                        "economic_exposure_mmi_10":
                            pager_data[
                                "economic_exposure_mmi_10"
                            ]
                    }
                ]
            )

            # Stage 1
            has_loss = (
                economic_loss_classifier.predict(
                    economic_features
                )[0]
            )

            # Stage 2
            if has_loss == 0:

                economic_prediction = 0.0

            else:

                prediction_log = (
                    economic_loss_regressor.predict(
                        economic_features
                    )[0]
                )

                economic_prediction = float(
                    np.expm1(
                        prediction_log
                    )
                )

                economic_prediction = max(
                    0.0,
                    economic_prediction
                )

            st.session_state.economic_prediction = (
                economic_prediction
            )

        except Exception as error:

            st.session_state.economic_prediction = None
            st.session_state.pager_data = None

            st.error(
                f"Impossible de charger les données "
                f"USGS PAGER : {error}"
            )


    # --------------------------------------------------------
    # ECONOMIC RESULT
    # --------------------------------------------------------

    if (
        st.session_state.economic_prediction
        is not None
        and
        st.session_state.pager_data
        is not None
    ):

        pager_data = (
            st.session_state.pager_data
        )

        economic_prediction = (
            st.session_state.economic_prediction
        )

        st.success(
            "Données USGS PAGER chargées avec succès."
        )

        st.subheader(
            "2. Informations du séisme"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Magnitude",
            pager_data["mag"]
        )

        col2.metric(
            "Profondeur",
            f'{pager_data["depth"]:.1f} km'
        )

        col3.metric(
            "MMI maximale",
            pager_data["maxmmi"]
        )

        st.write(
            f'Latitude : '
            f'{pager_data["latitude"]:.4f}'
        )

        st.write(
            f'Longitude : '
            f'{pager_data["longitude"]:.4f}'
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "3. Résultat du modèle"
        )

        st.metric(
            "Pertes économiques estimées",
            f"{economic_prediction:,.2f} $ US"
        )

        st.caption(
            "Estimation produite par le modèle V2 "
            "entraîné à partir des estimations "
            "économiques USGS PAGER."
        )


        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        st.subheader(
            "4. Localisation du séisme"
        )

        economic_map = folium.Map(
            location=[
                pager_data["latitude"],
                pager_data["longitude"]
            ],
            zoom_start=6
        )

        folium.CircleMarker(
            location=[
                pager_data["latitude"],
                pager_data["longitude"]
            ],
            radius=15,
            color="purple",
            fill=True,
            fill_color="purple",
            fill_opacity=0.8,
            tooltip=(
                f"Pertes estimées : "
                f"{economic_prediction:,.2f} $ US"
            ),
            popup=folium.Popup(
                f"""
                <b>USGS Event ID :</b>
                {pager_data["event_id"]}<br>

                <b>Magnitude :</b>
                {pager_data["mag"]}<br>

                <b>Profondeur :</b>
                {pager_data["depth"]:.1f} km<br>

                <b>MMI maximale :</b>
                {pager_data["maxmmi"]}<br>

                <b>Pertes estimées :</b>
                {economic_prediction:,.2f} $ US
                """,
                max_width=350
            )
        ).add_to(economic_map)

        st_folium(
            economic_map,
            width=None,
            height=500,
            key="economic_result_map",
            returned_objects=[]
        )