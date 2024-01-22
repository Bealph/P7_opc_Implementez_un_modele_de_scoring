import requests
import pandas as pd

base_url = "http://127.0.0.1:5000"

# Chargeons les données
df = pd.read_csv("top_50_train.csv", encoding='utf-8')
df.set_index('SK_ID_CURR', inplace=True)

num_client = df.index.unique()

def test_predict_endpoint():
    # Utilisez la route correcte ici, qui est '/api/infos_client/'
    response = requests.post(f'{base_url}/api/infos_client/', json={'input': num_client[0]})
    assert response.status_code == 200
    assert 'proba' in response.json()
    assert 'feature_names' in response.json()
    assert 'feature_importance' in response.json()
