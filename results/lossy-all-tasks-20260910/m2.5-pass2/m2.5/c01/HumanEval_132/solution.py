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
    opening = 0
    for i, ch in enumerate(string):
        if ch == '[':
            opening += 1
            if opening == 2:
                closing = 0
                for ch2 in string[i + 1:]:
                    if ch2 == ']':
                        closing += 1
                        if closing == 2:
                            return True
                return False
    return False
