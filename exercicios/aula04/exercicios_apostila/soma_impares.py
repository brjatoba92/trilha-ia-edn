n = int(input()) #numeros decasos

for _ in range(n): #valor da variavel, mantem o tipo e descarta a variavel
    x, y = map(int, input().split()) #dois numeros inteiros separand-os pelo espaço
    if x>y:
        x,y = y,x
    soma = sum(i for i in range(x+1, y) if i%2 != 0) 
    print(soma)