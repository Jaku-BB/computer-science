def get_natural_number_sum(number):
    return number > 1 and number + get_natural_number_sum(number - 1) or 1


print(get_natural_number_sum(5))
