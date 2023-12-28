import streamlit as st
import pandas as pd
from threading import Thread

# Chargement des données (à remplacer par votre logique de chargement de données)
data = pd.read_csv('votre_fichier.csv')

# Fonction pour le dashboard Streamlit
def streamlit_code():
    st.title("Dashboard de Prédictions")
    st.write("Ce dashboard permet d'accéder aux informations des clients et d'envoyer les prédictions à l'application.")

    # Dropdown pour sélectionner l'ID du client
    selected_id = st.selectbox("Sélectionnez l'ID du client", data['id_client'].unique())

    # Affichage des informations du client sélectionné sous forme de tableau (10 variables)
    if selected_id is not None:
        client_info = data[data['id_client'] == selected_id].head(1)  # Récupération des informations du client
        st.write("Informations du client sélectionné :")
        st.write(client_info)

    # Bouton Submit pour envoyer les données et probabilités à l'application
    if st.button('Submit'):
        # Logique pour calculer les probabilités (remplacez par votre logique de prédiction)
        proba_predictions = [0.75, 0.25, 0.8]  # Exemple de probabilités
        # Envoi des probabilités à l'application (à remplacer par votre logique d'envoi)
        # send_data_to_app(selected_id, proba_predictions)
        
        # Affichage des probabilités prédites
        st.write("Probabilités prédites :")
        st.write(proba_predictions)

# Fonction pour exécuter le dashboard Streamlit dans un thread
def streamlit_thread():
    st.button("Lancer l'application")
    thread = Thread(target=streamlit_code)
    thread.start()

if __name__ == '__main__':
    streamlit_thread()





'''
redesign du dash (image, texte d'expli, en quoi va concerner le dash, 5 a 6 ligne,  un dropdown(menu : id des clients, qui va permettre qd on clique sur l'id, ça affiche sous forme de tableau (max: 10 variables les plus important)
 les info du client), un bouton submit (qd on clique ça puisse envoyer les proba, les données à l'app, créer une fonction, en utilisant comme parametre les id client), et afficher dans le dashboard la liste des proba predict)


'''