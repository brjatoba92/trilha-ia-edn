"""
sequencia fibonnaci 11235
"""
import sympy as sp

# Definir a variável para a sequência
n = sp.symbols('n')

# Definir a sequência de Fibonacci usando SymPy
fib = sp.fibonacci(n)

# Exibir os primeiros 10 valores da sequência de Fibonacci
for i in range(10):
    print(f'Fibonacci({i}) =', fib.subs(n, i))

