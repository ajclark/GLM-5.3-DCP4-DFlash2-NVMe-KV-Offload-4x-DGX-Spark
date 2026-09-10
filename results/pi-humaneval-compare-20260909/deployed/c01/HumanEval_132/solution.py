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
    closing_bracket_index = []
    for i, char in enumerate(string):
        if char == '[':
            opening_bracket_index.append(i)
        else:
            closing_bracket_index.append(i)
    if len(opening_bracket_index) < 2 or len(closing_bracket_index) < 2:
        return False
    return opening_bracket_index[1] < closing_bracket_index[-2]
