import re

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = str(t)
    t = BR_RE.sub(" ", t)          # remove <br>, <br/>, <br />
    t = t.replace("&quot;", '"')   # light HTML entity cleanup
    t = t.replace("&amp;", "&")
    t = WS_RE.sub(" ", t).strip()  # normalize whitespace
    return t
