from convertson_idade import converson_idade

dias = int(input('Digite a idade em dias: '))

anos, meses, dias = converson_idade(dias)

print(f'{anos} ano(s)')
print(f'{meses} meses')
print(f'{dias} dias')