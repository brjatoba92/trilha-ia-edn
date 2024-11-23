import requests

response = requests.get('https://pokeapi.co/api/v2/pokemon/ditto') #link pokeapi
if response.status_code == 200:
    print(response.json(), 'name')