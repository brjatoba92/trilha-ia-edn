import json

response = '{"nome": "Picachu", "tipo": "Eletrico"}'

dados = json.loads(response)
print(dados['tipo'])