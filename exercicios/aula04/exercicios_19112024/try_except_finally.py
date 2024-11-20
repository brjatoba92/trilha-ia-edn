try:
    arquivo = open('dados.txt', 'r')
    print(arquivo.read())
except FileNotFoundError:
    print('Arquivo não encontrado')
finally:
    print('Encerrando a operação')
    if 'arquivos' in locals():
        arquivo.close()