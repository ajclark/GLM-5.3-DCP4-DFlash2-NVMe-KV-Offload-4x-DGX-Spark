def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less then 100. 
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """
    if a < 2:
        return False
    count = 0
    d = 2
    while d * d <= a:
        while a % d == 0:
            a //= d
            count += 1
        d += 1
    if a > 1:
        count += 1
    return count == 3
