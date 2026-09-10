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
    count = 0
    nested_close = False
    for ch in string:
        if ch == '[':
            count += 1
        elif ch == ']':
            if nested_close:
                return True
            if count > 0:
                count -= 1
            if count >= 1:
                nested_close = True
    return False
