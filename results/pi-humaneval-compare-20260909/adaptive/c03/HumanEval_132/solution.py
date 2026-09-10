def is_nested(string):
    '''
    Create a function that takes a string as input which contains only square brackets.
    The function should return True if and only if there is a valid subsequence of brackets 
    where at least one bracket in the subsequence is nested.

    is_nested('[[]]') ➞ True
    is_nested('[]]]]]]][[[[[]') ➞ False
    is_nested('[][]') ➞ False
    is_nested('[]') ➞ False
    is_nested('[[][]]') ➞ True
    is_nested('[[]][[') ➞ True
    '''
    n = len(string)
    open_before = [0] * (n + 1)
    close_after = [0] * (n + 1)
    for i in range(n):
        open_before[i + 1] = open_before[i] + (1 if string[i] == '[' else 0)
    for i in range(n - 1, -1, -1):
        close_after[i] = close_after[i + 1] + (1 if string[i] == ']' else 0)
    for j in range(n):
        if string[j] == '[' and open_before[j] >= 1 and close_after[j + 1] >= 2:
            return True
    return False
