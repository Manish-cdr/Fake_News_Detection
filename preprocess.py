import re

_DATELINE = re.compile(r"^.{0,80}?\(Reuters\)\s*-?\s*", re.S)
_SOURCE = re.compile(r"(?i)reuters")


def clean_text(s: str) -> str:
    """Remove source give-aways (Reuters dateline/name) so the model learns
    writing style and content instead of a publisher watermark."""
    s = _DATELINE.sub("", s)
    return _SOURCE.sub("", s)


def preprocess(s: str) -> str:
    return clean_text(s).lower()
