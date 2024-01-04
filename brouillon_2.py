import pandas as pd
import requests

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
# df.set_index('SK_ID_CURR', inplace=True) # Commentez cette ligne pour éviter l'indexation par SK_ID_CURR

# Define headers specifying the content type as JSON
headers = {
    "Content-Type": "application/json",
    # You might include other headers if needed, such as authorization headers, etc.
}

def get_infos_clients(df):
    all_clients_info = {}  # Pour stocker les informations de tous les clients

    for client_id in df['SK_ID_CURR'].unique():  # Utilisez la colonne SK_ID_CURR directement
        selected_client = df[df['SK_ID_CURR'] == client_id].drop('SK_ID_CURR', axis=1)  # Exclure la colonne SK_ID_CURR
        client_data = selected_client.to_dict(orient='records')[0]  # Utilisez orient='records' pour créer un dictionnaire

        try:
            response = requests.post("http://127.0.0.1:5000/api/infos_client", headers=headers, json=client_data)

            if response.status_code == 200:
                result = response.json()
                prediction_proba = result['proba']
                feature_importance = result['vars']

                #all_clients_info[client_id] = {'proba': prediction_proba, 'vars': feature_importance}

                return prediction_proba, feature_importance
                
            else:
                print(f"La requête pour le client {client_id} a échoué avec le code:", response.status_code, response.text)

        except requests.exceptions.RequestException as e:
            print(f"Une erreur s'est produite pour le client {client_id}:", e)

    return all_clients_info

# Collecter les informations pour tous les clients
all_clients_information = get_infos_clients(df)
