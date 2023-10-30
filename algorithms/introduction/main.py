def get_natural_number_sum(number):
    return number > 1 and number + get_natural_number_sum(number - 1) or 1


def main():
    number = int(input("Liczba naturalna: "))
    print(get_natural_number_sum(number))


if __name__ == '__main__':
    main()
