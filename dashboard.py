import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

from PIL import Image

import request_app as ra
import pickle
import shap

import matplotlib.pyplot as plt
import plotly.express as px

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

    st.write("Ce graphique vise à illustrer visuellement l'évaluation des demandes de crédit pour des individus sans ou avec peu d'historique de prêt, en mettant en lumière les 10 variables numériques les plus influentes."
              " Les barres plus hautes indiquent des variables plus déterminantes, fournissant ainsi une vue d'ensemble rapide des facteurs clés pris en compte dans la décision d'accorder un crédit.")
    st.write(" ")

    # Créer un graphique à barres avec Altair
    bar_plot = alt.Chart(top_10_df).mark_bar().encode(
        x=alt.X('Row', title='Variables', sort='-y'),  # Tri décroissant
        y=alt.Y('Importance', title='Importance')
    ).properties(
        width=500,
        height=400
    )

    # Afficher le graphique à barres
    st.altair_chart(bar_plot)


    st.write('------------------------------')

    ####################### Boxplot ###########################     

    st.write("Ce graphique combiné facilite une évaluation rapide et efficace de la situation des demandeurs de crédit"
             " Les points représentent les valeurs des variables numériques pour ce client, offrant une comparaison visuelle immédiate avec la distribution générale des données")
    
    st.write(" ")

    # Créer un DataFrame avec les noms de variables et leurs importances 
    feature_info = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Trier le DataFrame par l'importance dans l'ordre décroissant
    feature_info = feature_info.sort_values(by='Importance', ascending=False)

    # Sélectionner les 10 premières variables
    top_10_features = feature_info.head(10)

    # Afficher le top 10 des variables numériques dans client_data
    top_10_features_names = top_10_features['Feature'].tolist()
    top_10_client_data = client_data[top_10_features_names]

    # Filtrer les features du client pour inclure uniquement celles présentes dans top_10_features_names
    filtered_client_data = data_by_client[data_by_client.index.isin(top_10_features_names)]

    # Créer un DataFrame séparé pour les points du client
    client_points = pd.DataFrame({
        'Feature': filtered_client_data.index,
        'Value': filtered_client_data.values,
        'Type': [f"Client : {selected_client}"] * len(filtered_client_data)
    })

    
    boxplot = alt.Chart(top_10_client_data.melt()).mark_boxplot().encode(
        x='variable:O',
        y='value:Q'
    ).properties(
        width=600,
        height=400,
    )

    
    boxplot_by_client = alt.Chart(pd.concat([top_10_client_data, client_points])).mark_point(
        color='red',  # Couleur des points
        size=100,  # Taille des points
        filled=True
    ).encode(
        x='Feature:N',
        y='Value:Q',
        shape='Type:N'
    ).properties(
        width=600,
        height=400,
    )


    # Afficher les points rouges et le boxplot dans le même graphique
    combined_chart = boxplot + boxplot_by_client
    st.altair_chart(combined_chart, use_container_width=True)


    # Afficher la ligne du DataFrame du client dont l'Id est sélectionné
    st.write("Données du client sélectionné:")

    df_filtered_client_data = pd.DataFrame({'Value': filtered_client_data.values}, index=filtered_client_data.index)
    df_filtered_client_data.index.name = 'Feature'

    st.dataframe(df_filtered_client_data.reset_index().transpose())

    st.write('------------------------------')

    ###################### chartplot #####################
    
    st.write("À travers cette représentation graphique, nous sommes en mesure de présenter de manière claire et concise les facteurs qui ont un impact significatif sur la décision d'octroi de crédit."
             " Les caractéristiques, telles que définies dans le graphique, sont des éléments clés pris en compte dans l'évaluation du profil financier des demandeurs."
             " La position et la longueur de chaque barre reflètent respectivement l'importance de chaque caractéristique et son impact sur la décision d'accorder un crédit.")

    #st.write(" ")

    fig_chartplot = go.Figure(go.Bar(y=feature_names, x=feature_importance, orientation='h', marker_color='skyblue'))
    fig_chartplot.update_layout(xaxis_title='Importance', yaxis_title='Caractéristiques')

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig_chartplot)

    st.write('------------------------------')

    #####################  Decision Plot  #####################

    st.write("Ce graphique de Decision illustre l'importance relative de chaque caractéristique dans l'octroi du crédit."
             " À travers ce graphique interactif, chaque barre représente une caractéristique spécifique, tandis que la"
              " hauteur de la barre indique l'importance de cette caractéristique dans la décision d'octroi de crédit. "
              " Les caractéristiques qui contribuent le plus à l'accord de crédit sont visualisées par des barres plus hautes, tandis que celles ayant une influence moindre sont représentées par des barres plus courtes.")

    #st.write(" ")

    # Créer un graphique de décision avec plotly
    fig = go.Figure(go.Bar(x=feature_names, y=feature_importance, 
                               hoverinfo='x+y', marker=dict(color='skyblue')))
    fig.update_layout(xaxis_title='Caractéristiques', yaxis_title='Importance')
        
    st.plotly_chart(fig)

#-------------                --------------------
st.set_option('deprecation.showPyplotGlobalUse', False)  # Désactiver l'avertissement de dépréciation


##################################################################################################################################################################################################################################


if show_predictions:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    # Sélectionner les données du client choisi
    client_row = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0]
    client_data_filtered = client_row.drop(labels='SK_ID_CURR')

    # Filtrer les données pour obtenir les infos du client sélectionné
    client_data_without_label = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].drop(labels='SK_ID_CURR')

    ######################### Print PREDICTION ###########################


    # Afficher d'autres informations sur le client
    prediction_proba, feature_names, feature_importance = ra.get_infos_client(client_data_without_label)
    st.write("Les informations ci-dessous représentent la probabilité de prédiction associée au client sélectionné. Cette probabilité est calculée en utilisant" 
             " des caractéristiques spécifiques associées au profil du client. Plus la probabilité est élevée, plus le modèle considère que le client peut présenter "
               "certains comportements ou caractéristiques prédéfinis.") 
    
    # Assurez-vous que prediction_proba est une liste ou un tableau NumPy de valeurs numériques
    prediction_proba = np.array(prediction_proba, dtype=float)

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

    fig = go.Figure(data=[go.Pie(labels=classes, values=values, hole=.4)])

    st.write("Le graphique ci-dessous permet de visualiser l'accord de crédit pour des personnes avec peu ou pas d'historique de prêt. En un coup d'œil, il est possible"
             " d'évaluer visuellement la probabilité d'approbation ou de refus d'octroi de crédits à un client.")
    
    st.plotly_chart(fig)

    st.write('------------------------------')

    ######################## ChartPlot ##############################
    # Graphique ChartPlot
    fig_chartplot = px.bar(x=classes, y=values, labels={'x': 'Classe', 'y': 'Probabilité'},
                        title='ChartPlot des probabilités par classe predicte')
    st.plotly_chart(fig_chartplot)

    st.write('------------------------------')

    ######################## Decision Plot ##############################
    # Graphique Decision Plot
    
    #utiliser les données excepted_value et suivre la meme demarche que celle suivi pour shap



    ######################### SHAP ###########################

    # Récupérer le numéro de ligne dans le numpy array
    #client_index = client_data[client_data['SK_ID_CURR'] == selected_client].index[0]

    # Filtrer la dataframe pour ne conserver que les variables d'importance (feature_importance)
    prediction_proba, feature_names, feature_importance = ra.get_infos_client(client_data_without_label)

    # Convertir les noms de variables feature_names en majuscules
    feature_names_upper = [name.upper() for name in feature_names]

    top_10_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:11]
    top_10_features = [(feature_names_upper[i], feature_importance[i]) for i in top_10_indices]

    # Créer un DataFrame avec les 10 variables les plus importantes
    top_10_df = pd.DataFrame(top_10_features, columns=['Variables', 'Importance'])

    top_10_df['Variables'] = top_10_df['Variables'].str.lower()

    var_list = top_10_df['Variables'].tolist()

    # Filtrer le DataFrame client_data pour ne conserver que les colonnes d'importance
    filtered_feature = client_data[var_list]

    # Sélectionner les shap_values pour le client spécifié
    shap_values_client = shap_values[filtered_feature.index[0]]

    # Créer une dataframe pour les valeurs SHAP
    shap_df = pd.DataFrame(shap_values_client, columns=filtered_feature)


    # Récupérer les index 
    #filtered_feature_index = filtered_feature.index

    print(shap_df.shape)

    print( )

    
    print(shap_df)

    





    

   





    


   
    






# Afficher le bouton "Fermer l'affichage" s'il y a quelque chose à fermer
if show_close_button:
    if st.sidebar.button("Fermer l'affichage"):
        # Réinitialisation de l'affichage à l'étape initiale
        st.sidebar.markdown('<style>#variables_button, #predictions_button {display: block;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.header, .centered-text, img {display: block;}</style>', unsafe_allow_html=True)