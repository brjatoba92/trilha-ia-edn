# Faça um Programa que calcule a área de um quadrado, em seguida mostre o dobro desta área para o usuário.

lado_do_quadrado = float(input("Informe a aresta do quadrado, em metros: "))

area_do_quadrado = lado_do_quadrado ** 2

print(f'O dobro da area do quadrado é igual a {2 * area_do_quadrado} m')