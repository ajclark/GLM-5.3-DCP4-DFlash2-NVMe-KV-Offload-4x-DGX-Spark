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
    i = string.find('[')
    if i == -1:
        return False
    j = string.find('[', i + 1)
    if j == -1:
        return False
    k = string.find(']', j + 1)
    if k == -1:
        return False
    l = string.find(']', k + 1)
    return l != -1
