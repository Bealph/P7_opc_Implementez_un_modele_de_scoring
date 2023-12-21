import pandas as pd
import requests

# URL de mon API Flask
url = 'http://127.0.0.1:8200'  

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)

num_client = df.index.unique()

# Sélectionnez un client (par exemple, prenons le premier client dans cet exemple)
client_data = df.loc[num_client[0]].to_dict()

# Envoi de la requête HTTP au serveur Flask
try:
    response = requests.post(url, json=client_data)

    print(response)

    # Vérification de la réponse
    if response.status_code == 200:
        result = response.json()
        print("Prédictions de probabilité pour le client:", result['client_info'])
        print("Probabilités prédites:", result['prediction_proba'])
    else:
        print("La requête a échoué avec le code:", response.status_code)

except requests.exceptions.RequestException as e:
    print("Une erreur s'est produite lors de l'envoi de la requête:", e)
