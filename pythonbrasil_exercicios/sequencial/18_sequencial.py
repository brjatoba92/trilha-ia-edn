"""
Faça um programa que peça o tamanho de um arquivo para download (em MB) 
e a velocidade de um link de Internet (em Mbps), calcule e 
informe o tempo aproximado de download do arquivo usando este link (em minutos).
"""

tamanho_arquivo = float(input("Informe o tamanho do arquivo para download, em MB: "))
velocidade_internet = float(input("Informe a velocidade de download, em Mbps"))

tempo_minutos = (tamanho_arquivo / velocidade_internet) / 60

print(f'O arquivo de {tamanho_arquivo} MB a uma taxa de {velocidade_internet} Mbps levará {tempo_minutos:.2f} min para ser feito o download')
