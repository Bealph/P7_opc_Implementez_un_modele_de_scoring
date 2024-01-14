import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

from PIL import Image

import request_app as ra
import pickle
import shap

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go


#---------------------------------
# import des  shap_values
with open('shap_values.pkl', 'rb') as shap_file:
    shap_values = pickle.load(shap_file)

#---------------------------------
# import des excepted_value
with open('expected_value.pkl', 'rb') as excepted_value_file:
    excepted_value = pickle.load(excepted_value_file)

#--------------------------------------------
# Charger l'image
image = Image.open('image_app.jpeg')

# Afficher une colonne pour centrer l'image
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(image, width=250, output_format="JPEG")

# ----------------------------
# le body
    
st.markdown(
    """
<style>
.header {
    font-size : 24px;
    text-align : center;
    text-decoration : underline;
}
</style>

<style>
.centered-text {
    text-align : justify;
    text-align-last : center;
}
</style>

""",
unsafe_allow_html=True
)


st.markdown(
    '<h1 class="header">Tableau de bord en temps réel</h1>',
    unsafe_allow_html=True
)

# costomisation texte d'explication de l'application
st.markdown(
    '<p class="centered-text">Cette application vise à fournir un système de scoring avancé permettant\
        l\'accès aux informations essentielles des clients. Grâce à des modèles prédictifs, elle offre la possibilité\
            de générer des predictions pertinentes en temps réel.<br> <br> L\'objectif principale est de faciliter \
                la prise de décision en fournissant des analysées détaillées basées sur les données des clients.</p>',
    unsafe_allow_html=True
)

# --------------------------------------------

# Charger les données depuis votre fichier CSV
client_data = pd.read_csv("top_50_train.csv", encoding='utf-8')
descrip_columns = pd.read_csv('HomeCredit_columns_description.csv', encoding='ISO-8859-1')

lexique = descrip_columns[['Row', 'Description']]

# Menu déroulant pour sélectionner un client
selected_client = st.sidebar.selectbox("Sélectionnez un client", client_data['SK_ID_CURR'])

# Variables de contrôle pour afficher ou non les éléments
show_variables = False
show_predictions = False
show_close_button = False

# Si un client est sélectionné, afficher les boutons correspondants
if selected_client:
    # Afficher les boutons et récupérer leur état
    show_variables = st.sidebar.button("Afficher les 10 variables importantes", key="variables_button")
    show_predictions = st.sidebar.button("Affichage des prédictions probables", key="predictions_button")

    # Gestion de l'affichage des boutons
    if show_variables:
        show_close_button = True

    if show_predictions:
        show_close_button = True



# Affichage des éléments dans la colonne principale en fonction des boutons cliqués
if show_variables:

    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    data_by_client = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].drop(labels='SK_ID_CURR')

    prediction_proba, feature_names, feature_importance = ra.get_infos_client(data_by_client)

    # Convertir les noms de variables feature_names en majuscules
    feature_names_upper = [name.upper() for name in feature_names]

    top_10_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:11]
    top_10_features = [(feature_names_upper[i], feature_importance[i]) for i in top_10_indices]

    # Créer un DataFrame avec les 10 variables les plus importantes
    top_10_df = pd.DataFrame(top_10_features, columns=['Row', 'Importance'])

    # Fusionner avec le lexique pour obtenir les descriptions correspondantes
    top_10_lexique = pd.merge(top_10_df, lexique, on='Row')

    # Afficher les 10 variables les plus importantes et leurs descriptions
    st.write("Les 10 variables les plus importantes :")
    st.table(top_10_lexique[['Row', 'Description']].rename(columns={'Row': 'Variables', 'Description': 'Lexique'}))

    st.write('------------------------------')

    ####################### Histrogramme ###########################

    st.write("Observations des 10 variables les plus importantes :")

    # Trier le DataFrame par Importances de manière décroissante
    top_10_df = top_10_df.sort_values(by='Importance', ascending=False)

    bar_plot = alt.Chart(top_10_df).mark_bar().encode(
        x=alt.X('Row', title='Variables'),
        y=alt.Y('Importance', title='Importance')
    ).properties(
        width=500,
        height=500
    )

    st.altair_chart(bar_plot)

    st.write('------------------------------')

    ####################### Boxplot ###########################     # Placer les points rouges correspondant aux valeurs du client

    # Sélectionner les noms des 4 variables les plus importantes
    top_feature_names = feature_names[:4]  # des variables continues à prendre

    

    # Créer un DataFrame avec les 4 variables importantes pour le client sélectionné
    client_top_features = pd.DataFrame({name: [client_data[name]] for name in top_feature_names})

    client_top_features_by_client = pd.DataFrame({name: [data_by_client[name]] for name in top_feature_names})


    # Transformer le DataFrame pour qu'il soit en format long (nécessaire pour Altair)
    client_top_features = client_top_features.T.reset_index()

    client_top_features_by_client = client_top_features_by_client.T.reset_index()

    client_top_features.columns = ['variable', 'valeur']

    client_top_features_by_client.columns = ['variable', 'valeur']

    #print(client_data.head())

    

    st.write('------------------------------')

    #####################  Decision Plot  #####################

    # Créer un graphique de décision avec plotly
    fig = go.Figure(go.Bar(x=feature_names, y=feature_importance, 
                               hoverinfo='x+y', marker=dict(color='skyblue')))
    fig.update_layout(title='Importance des caractéristiques dans la prédiction',
                          xaxis_title='Caractéristiques', yaxis_title='Importance')
        
    st.plotly_chart(fig)

    ###################### chartplot #####################
    # Créer un graphique chartplot
    # Créer un graphique chartplot avec Plotly
    fig_chartplot = go.Figure(go.Bar(y=feature_names, x=feature_importance, orientation='h', marker_color='skyblue'))
    fig_chartplot.update_layout(title='Importance des caractéristiques pour la prédiction',
                                xaxis_title='Importance', yaxis_title='Caractéristiques')

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_chartplot)

    

    # Créer le boxplot avec Altair
    boxplot = alt.Chart(client_data).mark_boxplot().encode(
        x='variable:N',
        y='valeur:Q'
    ).properties(
        title='Boxplot des variables importantes pour le client'
    )

    # Créer un DataFrame séparé pour les points du client
    client_points = client_top_features_by_client.copy()
    client_points['type'] = 'client'

   # Afficher le boxplot avec les points du client
    chart = alt.Chart(pd.concat([client_top_features_by_client, client_points])).mark_point(
        color='red',  # Couleur des points
        size=100,  # Taille des points
        filled=True
    ).encode(
        x='variable:N',
        y='value:Q',
        shape='type:N'
    )

    # Afficher le boxplot
    st.altair_chart(boxplot, use_container_width=True)


    
    
    
    

    




# Afficher le 4 top des variable un boxplot et un point rouge pour voir où se trouve le client, un histogramme




#-------------                --------------------
st.set_option('deprecation.showPyplotGlobalUse', False)  # Désactiver l'avertissement de dépréciation


if show_predictions:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    # Sélectionner les données du client choisi
    client_row = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0]
    client_data_filtered = client_row.drop(labels='SK_ID_CURR')

    # Filtrer les données pour obtenir les infos du client sélectionné
    client_data = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].drop(labels='SK_ID_CURR')

    ######################### Print PREDICTION ###########################


    # Afficher d'autres informations sur le client
    prediction_proba, feature_names, feature_importance = ra.get_infos_client(client_data)
    st.write("Les informations ci-dessous représentent la probabilité de prédiction associée au client sélectionné. Cette probabilité est calculée en utilisant" 
             " des caractéristiques spécifiques associées au profil du client. Plus la probabilité est élevée, plus le modèle considère que le client peut présenter "
               "certains comportements ou caractéristiques prédéfinis.")
    #st.write(prediction_proba)  

    # Classes correspondant aux prêts remboursés et non remboursés
    classes = ['Prêt Remboursé', 'Prêt Non Remboursé']

    # Valeurs de probabilité associées à chaque classe
    values = [prediction_proba[0], prediction_proba[1]]

    # Affichage des probabilités sous forme de tableau
    st.write("Probabilité de Remboursement et Non Remboursement :")
    table_data = {'Classes predictes': classes, 'Probabilités de remboursement': values}
    st.table(table_data)

    st.write('------------------------------')

        ################################ Donut Chart   ##############################

    # Créer un graphique Donut Chart avec Plotly
    labels = ['Prêt Remboursé', 'Prêt Non Remboursé']
    values = [prediction_proba[0], prediction_proba[1]] 

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    #fig.update_layout(title='Prédictions')

    st.write("Le graphique ci-dessous permet de visualiser l'accord de crédit pour des personnes avec peu ou pas d'historique de prêt. En un coup d'œil, il est possible"
             " d'évaluer visuellement la probabilité d'approbation ou de refus d'octroi de crédits à un client.")
    
    st.plotly_chart(fig)

    st.write('------------------------------')



    ######################### SHAP ###########################

    # Obtenez les valeurs SHAP pour le client sélectionné
    shap_values_client = shap_values[shap_values.index == client_row.name]

    # Tracer le graphique SHAP
    shap_df = pd.DataFrame(shap_values_client, columns=client_data_filtered.index)

    shap_df = shap_df.T
    shap_df['feature'] = shap_df.index.astype(str)  # Conversion des noms de colonnes en chaînes
    shap_df['shap_value'] = shap_df.iloc[:, 0]  # Sélectionner les valeurs SHAP (ou ajustez l'index approprié)
    shap_df = shap_df.reset_index(drop=True)

    chart = alt.Chart(shap_df).mark_bar().encode(
        x='shap_value',
        y=alt.Y('feature', sort='-x')
    ).properties(
        width=500,
        height=500
    )

    st.altair_chart(chart, use_container_width=True)


   
    






# Afficher le bouton "Fermer l'affichage" s'il y a quelque chose à fermer
if show_close_button:
    if st.sidebar.button("Fermer l'affichage"):
        # Réinitialisation de l'affichage à l'étape initiale
        st.sidebar.markdown('<style>#variables_button, #predictions_button {display: block;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.header, .centered-text, img {display: block;}</style>', unsafe_allow_html=True)