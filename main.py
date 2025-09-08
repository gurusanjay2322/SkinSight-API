import requests

key = "a22930046800e169c95a1eec72804a56"
lat = "13.086396"
lon = "80.285613"

url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=metric&appid={key}"

response = requests.get(url)
print(response.status_code)
print(response.json())
