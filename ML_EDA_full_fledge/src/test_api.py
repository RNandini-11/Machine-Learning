# Sending a POST request with JSON
import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [
        100000,
        30,
        0.7,
        5000,
        1,
        0,
        0,
        1,
        0,
        0,
        35,
        10,
        75000,
        0,
        1,
        0,
        0,
        0,
        15,
        2,
        1000,
        0,
        1,
        3,
        0,
        1
    ]
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.text)
