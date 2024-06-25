# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import folium
from streamlit_folium import folium_static
import streamlit as st
from PIL import Image
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import pydeck as pdk
import folium
from folium.plugins import HeatMap



import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Configuration de la page
st.set_page_config(page_title="Mon Projet de Data Science", layout='wide')

# Ajout de la navigation dans le sidebar nommé "Sommaire"
st.sidebar.title("Sommaire")
pages = ["Contexte du Projet", "Exploration des Données", "Analyse de Données", "Prédiction"]
page = st.sidebar.radio("Aller vers la page", pages)

def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode()

if page == "Contexte du Projet":
    image_path = r"C:\Users\HP\Documents\images.png"
    st.markdown(
        f"""
        <style>
        .container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .text {{
            width: 75%;
            text-align: justify;
        }}
        .image {{
            width: 20%;
            background-image: url('{get_image_as_base64(image_path)}');
            background-repeat: no-repeat;
            background-size: contain;
            height: 200px;
        }}
        </style>
        <div class="container">
            <div class="text">
                <h1>Optimisation de la QoS</h1>
                <p>La qualité du signal, souvent mesurée par des indicateurs tels que le RSRP, est cruciale pour assurer une expérience utilisateur optimale et une couverture réseau adéquate. Traditionnellement, cette qualité est évaluée par des drives tests, qui bien que précis, sont coûteux et consomment beaucoup de temps et de ressources. Notre application est conçue pour optimiser la qualité de service de nos opérateurs clients.</p>
            </div>
            <div class="image"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Exploration des Données":
    st.title("Exploration des Données")
    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=['xlsx'])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        data = pd.read_excel(uploaded_file)
        st.session_state.data = data  # stocker les données chargées dans session_state

        st.write('Aperçu du dataset:')
        st.dataframe(data.head())

        # Afficher les dimensions du dataset
        st.write('Dimensions du dataset:', data.shape)

        st.write('Valeurs manquantes par attribut:')
        st.table(data.isnull().sum())

        duplicates = data[data.duplicated()]
        st.write(f"Nombre de doublons: {duplicates.shape[0]}")
        if st.checkbox("Afficher les doublons"):
            st.dataframe(duplicates)
    else:
        st.write("Veuillez télécharger un fichier Excel.")

elif page == "Analyse de Données":
    st.title("Analyse de Données")
    
    if 'data' in st.session_state:
        data = st.session_state.data  # Utiliser les données chargées dans session_state

        # Vérifier que la colonne 'RSRP' est présente dans le DataFrame
        if 'RSRP' in data.columns:
            # Permettre à l'utilisateur de sélectionner une variable parmi celles disponibles, excluant 'RSRP'
            options = [col for col in data.columns if col != 'RSRP']
            selected_column = st.selectbox("Choisir une variable pour comparer avec RSRP", options)

            # Créer un bouton pour générer le graphique
            if st.button(f'Afficher la variation de RSRP avec {selected_column}'):
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[selected_column], y=data['RSRP'], ax=ax)
                ax.set_title(f"RSRP vs {selected_column}")
                ax.set_xlabel(selected_column)
                ax.set_ylabel('RSRP (dBm)')
                st.pyplot(fig)
        else:
            st.error("La colonne 'RSRP' est manquante dans vos données.")
    else:
        st.error("Aucune donnée n'est chargée. Veuillez télécharger un fichier Excel.")
    

elif page == "Prédiction":
    

    def load_data():
        # Remplacer ceci par le chargement des données réelles
        data = pd.read_excel(r"C:\Users\HP\Downloads\Book3.xlsx")
        return data

    # Fonction pour préparer les données
    def prepare_data(data):
        features = data[['Longitude', 'Latitude', 'MNC', 'CNX_SYS_BAND']]
        target = data['RSRP']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return features, target, X_train, X_test, y_train, y_test

    # Fonction pour entraîner le modèle
    def train_model(X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    # Fonction pour évaluer le modèle
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        return r2, mape, y_pred

    # Fonction pour classifier le RSRP et attribuer des couleurs
    def classify_rsrp(rsrp):
        if rsrp > -90:
            return 'Green'
        elif -105 <= rsrp <= -90:
            return 'Yellow'
        elif -120 <= rsrp < -106:
            return 'Orange'
        else:
            return 'Red'

    if st.button("Charger et prédire le RSRP"):
        st.title('Pérformance du modèle ')
        
        data = load_data()
        features, target, X_train, X_test, y_train, y_test = prepare_data(data)
        model = train_model(X_train, y_train)
        r2, mape, y_pred = evaluate_model(model, X_test, y_test)

        st.write(f"R² (Coefficient de détermination): {r2:.3f}")
        st.write(f"MAPE (Erreur moyenne absolue en pourcentage): {mape:.3f}")

        # Prédiction sur les nouvelles données
        data['RSRP'] = model.predict(data[['Longitude', 'Latitude', 'MNC', 'CNX_SYS_BAND']])
        data['Color'] = data['RSRP'].apply(classify_rsrp)

        # Création de la carte avec Folium
        m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=12)

        # Ajout des points de données à la carte
        for _, row in data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"RSRP: {row['RSRP']} dBm\nClassification: {row['Color']}",
                color=row['Color'].lower(),
                fill=True,
                fill_color=row['Color'].lower()
            ).add_to(m)

        # Affichage de la carte
        st.components.v1.html(m._repr_html_(), width=700, height=500)