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
    factors = 0
    n = a
    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
            factors += 1
        d += 1
    if n > 1:
        factors += 1
    return factors == 3
