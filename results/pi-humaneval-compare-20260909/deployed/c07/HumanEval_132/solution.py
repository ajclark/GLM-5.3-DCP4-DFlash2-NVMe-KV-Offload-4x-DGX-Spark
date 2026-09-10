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
    opens = 0
    closed_inner = False
    for ch in string:
        if ch == '[':
            opens += 1
        elif ch == ']':
            if closed_inner:
                return True
            if opens >= 2:
                closed_inner = True
            opens = max(opens - 1, 0)
    return False
