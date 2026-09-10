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
    closes = [i for i, ch in enumerate(string) if ch == ']']
    if len(closes) < 2:
        return False
    limit = closes[-2]
    opens = 0
    for ch in string[:limit]:
        if ch == '[':
            opens += 1
            if opens == 2:
                return True
    return False
