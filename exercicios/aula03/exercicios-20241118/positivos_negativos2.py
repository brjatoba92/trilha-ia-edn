#criar um valor sequencial que atinja o pedido no exercicio
for a in range(0, 101, 2):  # a deve ser par
    for b in range(0, 101):
        for c in range(b + 1, 101):  # c > b
            for d in range(a + 1, 101):  # d > a
                if b > c and d > a and (c + d) > (a + b) and a % 2 == 0:
                    print(f"Valores aceitos: a = {a}, b = {b}, c = {c}, d = {d}")
                else:
                    print(f"Valores não aceitos: a = {a}, b = {b}, c = {c}, d = {d}")



