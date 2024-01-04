import pandas as pd
import requests

# URL de mon API Flask
#url = "http://127.0.0.1:5000/api/predict_proba"

#url = "http://127.0.0.1:5000/api/client_data"



# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)

num_client = df.index.unique()

# Define headers specifying the content type as JSON
headers = {
    "Content-Type": "application/json",
    # You might include other headers if needed, such as authorization headers, etc.
}


# Sélectionnez un client (par exemple, prenons le premier client dans cet exemple)


# Envoi de la requête HTTP au serveur Flask

def get_infos_client(selected_client):
    client_data = selected_client.to_dict()

    try:
        response = requests.post("http://127.0.0.1:5000/api/infos_client", headers=headers, json=client_data)

        # Vérification de la réponse
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            prediction_proba = result['proba']
            feature_importance = result['vars']

            return prediction_proba, feature_importance
            
        else:
            print("La requête a échoué avec le code:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Une erreur s'est produite lors de l'envoi de la requête:", e)
