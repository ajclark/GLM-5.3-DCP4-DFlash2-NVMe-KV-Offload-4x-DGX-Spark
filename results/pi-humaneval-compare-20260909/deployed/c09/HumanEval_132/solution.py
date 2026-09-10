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
    # A valid nested subsequence exists iff the pattern '[[]]'
    # appears as a subsequence of the input string.
    pos = 0
    for ch in '[[]]':
        while pos < len(string) and string[pos] != ch:
            pos += 1
        if pos == len(string):
            return False
        pos += 1
    return True
