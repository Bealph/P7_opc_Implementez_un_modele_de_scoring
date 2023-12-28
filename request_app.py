import pandas as pd
import requests

# URL de mon API Flask
url = "http://127.0.0.1:5000/api/predict_proba"

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
client_data = df.iloc[0].to_dict()

# Envoi de la requête HTTP au serveur Flask
try:
    response = requests.post(url, headers=headers, json=client_data)

    # Vérification de la réponse
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        prediction_proba = result['proba']
        #print("Prédictions de probabilité pour le client:", result['client_info'])
        print("Probabilités prédites:", prediction_proba)
    else:
        print("La requête a échoué avec le code:", response.status_code, response.text)

except requests.exceptions.RequestException as e:
    print("Une erreur s'est produite lors de l'envoi de la requête:", e)


