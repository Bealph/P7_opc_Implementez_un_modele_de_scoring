import pandas as pd
import requests
import json

# URL de mon API Flask
url = 'http://127.0.0.1:5000/prediction'  

# Chargeons les données
df = pd.read_csv(r"C:/Users/Microsoft/Desktop/Dossiers formation/Formation OC/Data science/P7/Donnees projet/P7_opc_Implementez_un_modele_de_scoring/top_50_train.csv", encoding='utf-8')
df.drop(columns='TARGET', inplace=True)
num_client = df.SK_ID_CURR.unique()

# Sélectionnez un client (par exemple, prenons le premier client dans cet exemple)
client_data = df[df['SK_ID_CURR'] == num_client[0]].to_dict(orient='records')[0]

# Envoi de la requête HTTP au serveur Flask
response = requests.post(url, json=client_data)

# Vérification de la réponse
if response.status_code == 200:
    result = response.json()
    print("Prédictions de probabilité pour le client:", result['client_info'])
    print("Probabilités prédites:", result['prediction_proba'])
else:
    print("La requête a échoué avec le code:", response.status_code)
