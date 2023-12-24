import pandas as pd
import requests

# URL de mon API Flask
url = "http://127.0.0.1:5000"

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)

num_client = df.index.unique()

# Sélectionnez un client (par exemple, prenons le premier client dans cet exemple)
client_data = df.loc[num_client[0]].to_dict()

# Envoi de la requête HTTP au serveur Flask
try:
    response = requests.post(url, json=client_data)

    # Vérification de la réponse
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        prediction_proba = result['prediction_proba']
        print("Prédictions de probabilité pour le client:", result['client_info'])
        print("Probabilités prédites:", prediction_proba)
    else:
        print("La requête a échoué avec le code:", response.status_code, response.text)

except requests.exceptions.RequestException as e:
    print("Une erreur s'est produite lors de l'envoi de la requête:", e)


#--------------------------------------------------------
'''# Streamlit setup
def streamlit_thread():
    def streamlit_code():
        st.title("application Streamlit")

    sthread = Thread(target=streamlit_code)
    sthread.start()

if __name__ == '__main__':
    sthread = Thread(target=streamlit_thread)
    sthread.start()
    app.run(debug=True)'''