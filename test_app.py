import requests

base_url = "http://127.0.0.1:5000"

def test_predict_endpoint() :
    response = requests.post('{base_url}/api/predict_proba', json={'input': [6]})
    assert response.status_code == 200
    assert 'prediction' in response.json()
