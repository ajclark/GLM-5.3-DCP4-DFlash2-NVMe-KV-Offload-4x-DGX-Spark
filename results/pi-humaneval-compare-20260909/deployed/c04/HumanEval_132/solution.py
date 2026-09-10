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
    opening_bracket_index = []
    nested = False
    for i, ch in enumerate(string):
        if ch == '[':
            opening_bracket_index.append(i)
        else:
            if opening_bracket_index:
                if i - opening_bracket_index[-1] > 1:
                    nested = True
                opening_bracket_index.pop()
    return nested
