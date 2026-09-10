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
    found_nested_open = False
    for char in string:
        if char == '[':
            opens += 1
        elif char == ']':
            if found_nested_open:
                return True
            if opens >= 2:
                found_nested_open = True
            opens = max(opens - 1, 0)
    return False
