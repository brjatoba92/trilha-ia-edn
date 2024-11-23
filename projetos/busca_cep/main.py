from fastapi import FastAPI, HTTPException
import requests

app = FastAPI(title="API Consulta de CEP", description="API criada junto com os alunos da EDN para consulta de CEP")

@app.get("/")

def home():
    """
    ROTA INICIAL PARA VERIFICAR SE A API ESTA OK
    """
    return {"message": "Bem vindo a API de Consulta de CEP"}

@app.get("/consulta-cep/{cep}")
def consulta_cep(cep:  str):
    """
    consulta as informações do CEP e devolve o endereço
    cep no formato 00000000
    """
    
    if len(cep) != 8 or not cep.isdigit():
        raise HTTPException(status_code=400, detail='este cep esta invalido')
    response = requests.get(f'https://viacep.com.br/ws/{cep}/json/')
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail='o servidor esta fora do ar')
    
    data = response.json()

    if "erro" in data:
        raise HTTPException(status_code=404, detail='CEP NÃO ENCONTRADO')
    
    return data