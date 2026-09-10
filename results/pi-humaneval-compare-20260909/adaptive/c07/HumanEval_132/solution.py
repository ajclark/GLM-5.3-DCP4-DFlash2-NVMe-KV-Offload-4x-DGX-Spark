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
    state = 0
    for c in string:
        if c == '[':
            if state == 0:
                state = 1
            elif state == 1:
                state = 2
        else:
            if state == 2:
                state = 3
            elif state == 3:
                return True
    return False
