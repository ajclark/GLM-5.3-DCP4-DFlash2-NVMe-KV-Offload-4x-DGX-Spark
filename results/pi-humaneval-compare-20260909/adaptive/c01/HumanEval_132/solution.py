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
    opens = [i for i, c in enumerate(string) if c == '[']
    closes = [i for i, c in enumerate(string) if c == ']']
    if len(opens) < 2 or len(closes) < 2:
        return False
    second_open = opens[1]
    second_to_last_close = closes[-2]
    return second_to_last_close > second_open
