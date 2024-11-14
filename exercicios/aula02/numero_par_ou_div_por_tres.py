# enquanto o numero (index) estiver entre (1 e 100)
# se o numero for divisivel por dois com resto zero ou divisivel por tres com resto 0

for i in range(1,100):
    if i % 2 == 0 or i % 3 == 0: # pega somente a parte inteira desconsiderando o resto
        print(f'{i} divisivel por 2 ou por tres')
