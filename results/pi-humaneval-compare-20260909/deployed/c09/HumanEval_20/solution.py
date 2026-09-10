from typing import List, Tuple


def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each
    other and return them in order (smaller number, larger number).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
    (2.0, 2.0)
    """
    closest_pair = None
    distance = None
    for i, x in enumerate(numbers):
        for j, y in enumerate(numbers):
            if i < j:
                d = abs(x - y)
                if distance is None or d < distance:
                    distance = d
                    closest_pair = (min(x, y), max(x, y))
    return closest_pair
