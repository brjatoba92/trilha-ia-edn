def calcular_media_idades():
    """
    função que calcula a media de idades e não considera a negativa (break)
    """
    soma_idades = 0
    quantidade = 0

    while True:
        idade = int(input())
        if idade < 0:
            break
        soma_idades += idade
        quantidade += 1

    if quantidade > 0:
        media = soma_idades / quantidade
        return f'{media:.2f}'
    else:
        return 'Nenhuma idade válida inserida'

def main():
    try:
        media_idades = calcular_media_idades()
    except ValueError:
        print('Erro: entrada invalda, valor inserir outra entrada')

if __name__ == "__main__":
    main()