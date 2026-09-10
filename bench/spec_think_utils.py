"""Offline thinking metadata and visible-answer extraction; never tokenizes remotely."""
import re

WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
TOKEN_HEURISTIC = 'round(1.04 * Unicode word count); tokenizer-free estimate, not exact tokens'


def thinking_enabled(row):
    explicit = row.get('thinking')
    configured = row.get('comparison_settings', {}).get('thinking')
    for value in (explicit, configured):
        if value is not None and type(value) is not bool:
            raise ValueError('thinking metadata must be boolean')
    if explicit is not None and configured is not None and explicit != configured:
        raise ValueError('thinking metadata disagrees with configured request')
    return explicit if explicit is not None else configured


def completion_parts(text, reasoning='', *, thinking=False, reasoning_field_present=False):
    """Separate parser fields or raw tags, including a prompt-prefilled opening tag.

    A truncated open span has no visible answer. With a separate reasoning field
    the server has already removed the tags; plain content is the visible answer.
    """
    spans = re.findall(r'<think>(.*?)(?:</think>|$)', text, re.DOTALL)
    first_close, first_open = text.find('</think>'), text.find('<think>')
    prefilled = [text[:first_close]] if first_close >= 0 and (first_open < 0 or first_close < first_open) else []
    span_text = '\n'.join(prefilled + spans)
    if '</think>' in text:
        _, visible = text.rsplit('</think>', 1)
        # A further open span is not a completed visible answer.
        if '<think>' in visible:
            visible = ''
    elif '<think>' in text:
        visible = ''
    elif thinking and not reasoning and not reasoning_field_present:
        # Raw stream may omit the opening tag already present in the prompt.
        span_text, visible = text, ''
    else:
        span_text, visible = '', text
    return (reasoning or span_text), visible


def estimated_tokens(text):
    return round(1.04 * len(WORD.findall(text)))


def completion_accounting(text, reasoning='', usage=None, *, thinking=False, reasoning_field_present=False):
    reasoning_text, visible = completion_parts(text, reasoning, thinking=thinking,
                                              reasoning_field_present=reasoning_field_present)
    usage = usage or {}
    details = usage.get('completion_tokens_details') or {}
    exact = details.get('reasoning_tokens', usage.get('reasoning_tokens'))
    if exact is not None:
        if type(exact) is not int or exact < 0 or exact > usage.get('completion_tokens', exact):
            raise ValueError('invalid provider reasoning token count')
        count, source = exact, ('usage.completion_tokens_details.reasoning_tokens'
                                if 'reasoning_tokens' in details else 'usage.reasoning_tokens')
    else:
        count = estimated_tokens(reasoning_text)
        source = ('reasoning_field_word_estimate' if reasoning else
                  'think_span_word_estimate' if reasoning_text else 'no_reasoning_observed')
    return {'reasoning_tokens': count, 'reasoning_tokens_estimated': exact is None,
            'reasoning_tokens_source': source, 'reasoning_token_heuristic': TOKEN_HEURISTIC,
            'reasoning_words': len(WORD.findall(reasoning_text)),
            'visible_answer_words': len(WORD.findall(visible)),
            'visible_answer_tokens_estimate': estimated_tokens(visible)}
