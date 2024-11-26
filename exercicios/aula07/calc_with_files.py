def process_numbers_file(file_name):
    with open(file_name, 'r') as file:
        numbers = [int(line.strip()) for line in file.readlines()]

    total = sum(numbers) #Total
    average = total / len(numbers) #media
    min_value = min(numbers)
    max_value = max(numbers)

    return total, average, min_value, max_value

if __name__ == "__main__":
    file_name = input("Digite o nome do arquivo: ")
    total, average, min_value, max_value = process_numbers_file(file_name)

    print(f'Soma: {total}')
    print(f'Media: {average}')
    print(f'Maximo: {max_value}')
    print(f'Minimo: {min_value}')