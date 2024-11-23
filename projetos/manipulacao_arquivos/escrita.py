with open("escrita.txt","w") as file:
    file.write("Este é um exemplo de escrita em arquivo. \n")

with open("escrita.txt", "r") as file:
    content = file.read()
    print(content)