from time import time


def get_natural_number_sum(number):
    return number > 1 and number + get_natural_number_sum(number - 1) or 1


def main():
    number = int(input("Liczba naturalna: "))

    start_time = time()
    print(f"Wynik: {get_natural_number_sum(number)}\nCzas wykonania: {(time() - start_time) * 1000}")


if __name__ == '__main__':
    main()
