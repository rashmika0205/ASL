# grammar.py - turn a list of recognized signs (glosses) into natural sentences,
# joining related ideas with conjunctions (and, but, so, because, and then).
#
# Usage in predict.py:
#     from grammar import to_sentence
#     text = to_sentence(["I", "WANT", "WATER", "I", "WANT", "EAT"])
#     # -> "I want water and I want to eat."
#     speak(text)   # speak the corrected sentence instead of the raw word list
#
# How ideas are separated: sign I or YOU again to start a new idea.
#   I WANT WATER  I WANT EAT   ->  "I want water and I want to eat."
#   I BAD  I WANT HELP         ->  "I feel bad, so I want help."
# Questions and greetings (HELLO, BYE, THANK YOU) stay as their own sentences.
#
# Questions (WHERE / WHAT) work in either word order, like real ASL:
#   WHERE YOU HOME   /  YOU HOME WHERE   ->  "Where is your home?"
#   YOU WANT EAT WHAT                    ->  "What do you want to eat?"
#
# Two modes:
#   1. LLM (needs internet + `pip install anthropic` + ANTHROPIC_API_KEY env var)
#   2. Rule-based fallback (offline, deterministic) - used automatically if the LLM call fails.

SYSTEM = (
    "You convert ASL gloss sequences into natural, grammatical English (one sentence or a few "
    "short ones). The vocabulary is limited to: HELLO, BYE, THANK YOU, PLEASE, "
    "SORRY, I, YOU, YES, NO, WATER, EAT, TOILET, WHERE, GOOD, BAD, HOME, HELP, WANT, WHAT, NAME. "
    "Add only the minimal words needed (articles, 'am/is/are', 'to', possessives, question word order). "
    "When the glosses contain several related ideas (a new idea usually starts when I or YOU is "
    "signed again), join them into one sentence with a fitting conjunction: 'and', 'but', 'so', "
    "'because' or 'and then'. Keep questions and greetings as separate sentences. "
    "Do not invent new meaning or extra content. Treat 'YOU NAME WHAT' as 'What is your name?'. "
    "ASL puts question words at the start or the end, so treat 'WHERE YOU HOME' and "
    "'YOU HOME WHERE' both as 'Where is your home?'. "
    "A pronoun (I/YOU) next to a noun in a WHERE/WHAT question is a possessive: "
    "'WHERE I HOME' = 'Where is my home?'. "
    "More examples: 'WHERE YOU' = 'Where are you?'; 'WHERE EAT' = 'Where can I eat?'; "
    "'WHERE YOU WANT HOME' = 'Where do you want to go?'; "
    "'YOU WANT EAT WHAT' = 'What do you want to eat?'; 'I WANT YOU HELP' = 'I want you to help me.' "
    "Output ONLY the final text, nothing else."
)

SOCIAL = {"HELLO": "Hello!", "BYE": "Goodbye!", "THANK YOU": "Thank you!"}
VERBS = {"EAT"}   # "I WANT EAT" -> "I want to eat" (but "I WANT HELP" -> "I want help")
ADJ = {"GOOD", "BAD", "SORRY"}

SUBJECTS = {"I", "YOU"}
# words that make a clause "complete enough" that a new subject starts a NEW idea
PREDICATES = {"WANT", "EAT", "HELP", "WATER", "TOILET", "HOME", "GOOD", "BAD", "SORRY"}
# things a person asks for / needs
NEEDS = {"WANT", "HELP", "EAT", "TOILET", "WATER", "HOME"}
# a new idea never starts right after one of these (they still need their object)
NEEDS_OBJECT = {"WANT", "WHERE", "WHAT", "NAME"}

# nouns that can be "owned" in a question: WHERE YOU HOME -> "Where is your home?"
NOUNS = {"HOME": "home", "WATER": "water", "TOILET": "toilet"}
POSS = {"I": "my", "YOU": "your"}


def _where(c, please=False):
    """
    Build a WHERE question from the remaining glosses (WHERE already removed).
    Order does not matter: WHERE YOU HOME == YOU HOME WHERE.
    Returns None if no special pattern matches (caller falls back to generic).
    """
    subj = next((w for w in c if w in SUBJECTS), None)
    noun = next((w for w in c if w in NOUNS), None)
    s = "I" if subj == "I" else "you"

    # WHERE YOU WANT HOME / WHERE YOU WANT EAT
    if "WANT" in c:
        if "EAT" in c:
            return f"Where do {s} want to eat?"
        if "HOME" in c or "TOILET" in c:
            return f"Where do {s} want to go?"
        return None

    # WHERE YOU HOME -> Where is your home?   WHERE I WATER -> Where is my water?
    if noun and subj:
        return f"Where is {POSS[subj]} {NOUNS[noun]}?"

    # WHERE HOME / WHERE WATER / WHERE TOILET
    if noun:
        if noun == "HOME":
            return "Where is home?"
        return f"Where is the {NOUNS[noun]}?"

    if "EAT" in c:
        return "Where do you eat?" if subj == "YOU" else "Where can I eat?"
    if "HELP" in c:
        return "Where can I get help?"
    if subj == "YOU":
        return "Where are you?"
    if subj == "I":
        return "Where am I?"
    return None


def _what(c, tail=""):
    """Build a WHAT question from the remaining glosses (WHAT already removed)."""
    if "NAME" in c:
        who = "my" if ("I" in c and "YOU" not in c) else "your"
        return f"What is {who} name{tail}?"
    if "WANT" in c:
        if "EAT" in c:
            return f"What do you want to eat{tail}?"
        return f"What do you want{tail}?"
    if "EAT" in c:
        return f"What do you eat{tail}?"
    return "What?"


def _clause(c):
    """One idea (list of glosses) -> one sentence ending in '.' or '?'."""
    c = list(c)
    please = "PLEASE" in c
    c = [w for w in c if w != "PLEASE"]
    if not c:
        return "Please."
    if c in (["YES"], ["NO"]):
        return c[0].capitalize() + (", please." if please else ".")
    if c == ["SORRY"]:
        return "Sorry."

    q = None
    for qw in ("WHAT", "WHERE"):
        if qw in c:
            q = qw
            c = [w for w in c if w != qw]

    tail = ", please" if please else ""

    # ---- questions are handled by their own rules first ----
    if q == "WHAT":
        return _what(c, tail)
    if q == "WHERE":
        if not c:
            return "Where?"
        r = _where(c, please)
        if r:
            return r
        # generic fallback below

    out = []
    for i, w in enumerate(c):
        nxt = c[i + 1] if i + 1 < len(c) else None
        prev = c[i - 1] if i else None
        if w == "I":
            out.append("I")
        elif w == "YOU":
            if nxt == "NAME":
                out.append("your")
            elif prev == "WANT" and nxt == "HELP":
                out.append("you to")          # I want you to help me
            else:
                out.append("you")
        elif w == "NAME":
            out.append("name")
        elif w == "WANT":
            out.append("want to go" if nxt == "HOME"
                       else "want to" if nxt in VERBS or nxt == "TOILET"
                       else "want")
        elif w == "EAT":
            out.append("eat")
        elif w == "HELP":
            out.append("help" if prev == "WANT" else "help me")
        elif w == "WATER":
            out.append("water")
        elif w == "TOILET":
            out.append("use the toilet" if prev == "WANT"
                       else "need the toilet" if prev in SUBJECTS
                       else "the toilet")
        elif w == "HOME":
            out.append("go home" if prev in SUBJECTS else "home")
        elif w in ADJ:
            cop = {"I": "am", "YOU": "are"}.get(prev)
            if w == "BAD" and prev == "I":
                out.append("feel bad")
            elif cop:
                out.append(f"{cop} {w.lower()}")
            else:
                out.append(w.lower())
        elif w in ("YES", "NO"):
            out.append(w.capitalize() + ",")
    phrase = " ".join(out)

    if q == "WHERE":
        return f"Where is {phrase}{tail}?"

    if please and out and out[0] in ("help me", "eat"):
        return f"Please {phrase}."
    sentence = phrase[0].upper() + phrase[1:] if phrase else ""
    return f"{sentence}{tail}."


def _split_ideas(words):
    """
    Group a flat gloss list into units:
        ("social", "HELLO")          greetings / thanks (own sentence)
        ("idea", ["I", "WANT", ...]) one idea (clause)
    A new idea starts when I / YOU (or WHERE) shows up after a finished idea.
    """
    units, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            units.append(("idea", cur))
            cur = []

    for i, w in enumerate(words):
        if w in SOCIAL:
            flush()
            units.append(("social", w))
            continue
        # a WHERE with nothing after it (or only a greeting) closes the current
        # idea as its question word:  YOU HOME WHERE -> "Where is your home?"
        trailing_where = (w == "WHERE" and cur
                          and all(x in SOCIAL for x in words[i + 1:]))
        if trailing_where:
            cur.append(w)
            continue
        if w in SUBJECTS or w == "WHERE":
            finished = (any(x in PREDICATES for x in cur)
                        or any(x in ("WHAT", "WHERE") for x in cur))
            if cur and finished and cur[-1] not in NEEDS_OBJECT:
                flush()
        cur.append(w)
    flush()
    return units


def _conjunction(a, b):
    """
    Pick the word that links idea `a` to idea `b`.
    Returns (conjunction, comma_before_it).
    """
    sa, sb = set(a), set(b)
    if ("GOOD" in sa and "BAD" in sb) or ("BAD" in sa and "GOOD" in sb):
        return "but", True
    if "SORRY" in sa and sb & NEEDS:
        return "but", True
    if "BAD" in sb and sa & NEEDS:          # I want help because I feel bad
        return "because", False
    if "BAD" in sa and sb & NEEDS:          # I feel bad, so I want help
        return "so", True
    if "EAT" in sa and sb & {"HOME", "TOILET"}:   # I eat, and then I go home
        return "and then", True
    return "and", False


def _join(ideas):
    """Join one or more statement ideas into a single sentence."""
    text = _clause(ideas[0])[:-1]
    for prev, cur in zip(ideas, ideas[1:]):
        conj, comma = _conjunction(prev, cur)
        if text.endswith("please"):          # "..., please, and I want ..."
            comma = True
        nxt = _clause(cur)[:-1]
        if nxt != "I" and not nxt.startswith("I "):
            nxt = nxt[0].lower() + nxt[1:]
        text += ("," if comma else "") + f" {conj} {nxt}"
    return text + "."


def rule_based(words):
    sentences, run = [], []      # run = consecutive statements to join with conjunctions

    def flush_run():
        nonlocal run
        if run:
            sentences.append(_join(run))
            run = []

    for kind, item in _split_ideas(words):
        if kind == "social":
            flush_run()
            sentences.append(SOCIAL[item])
        elif _clause(item).endswith("?"):
            flush_run()
            sentences.append(_clause(item))
        else:
            run.append(item)
    flush_run()
    return " ".join(sentences)


def llm(words, model="claude-haiku-5-5"):
    import anthropic  # pip install anthropic
    # max_retries=0 so an offline demo falls back to the rule-based mode quickly
    client = anthropic.Anthropic(max_retries=0)  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model, max_tokens=150, system=SYSTEM,
        messages=[{"role": "user", "content": " ".join(words)}],
        timeout=3,
    )
    return msg.content[0].text.strip()


def to_sentence(words, use_llm=True):
    if not words:
        return ""
    if use_llm:
        try:
            return llm(words)
        except Exception:
            pass  # offline / no key / timeout -> fall back
    return rule_based(words)


if __name__ == "__main__":
    tests = [
        # single ideas (unchanged behaviour)
        ["HELLO", "YOU", "NAME", "WHAT"],
        ["I", "WANT", "WATER", "PLEASE"],
        ["WHERE", "TOILET"],
        ["I", "WANT", "EAT"],
        ["PLEASE", "HELP"],
        ["I", "SORRY"],
        ["YOU", "GOOD"],
        ["I", "WANT", "HOME"],
        ["THANK YOU", "BYE"],
        ["YES"],
        # several ideas joined with conjunctions
        ["I", "WANT", "WATER", "I", "WANT", "EAT"],
        ["I", "BAD", "I", "WANT", "HELP"],
        ["I", "WANT", "HELP", "I", "BAD"],
        ["YOU", "GOOD", "I", "BAD"],
        ["I", "EAT", "I", "HOME"],
        ["I", "EAT", "I", "WANT", "TOILET"],
        ["I", "SORRY", "I", "WANT", "HELP"],
        ["PLEASE", "HELP", "I", "BAD"],
        ["I", "WANT", "HOME", "I", "BAD"],
        ["HELLO", "I", "WANT", "WATER", "PLEASE", "I", "WANT", "EAT", "THANK YOU"],
        # statement + question
        ["I", "WANT", "WATER", "WHERE", "TOILET"],
        ["WHERE", "TOILET", "I", "WANT", "HELP"],
        ["I", "WANT", "WATER", "YOU", "NAME", "WHAT"],
        # NEW: more WHERE / WHAT questions (either word order)
        ["WHERE", "YOU", "HOME"],
        ["YOU", "HOME", "WHERE"],
        ["WHERE", "I", "HOME"],
        ["WHERE", "YOU", "WATER"],
        ["WHERE", "HOME"],
        ["WHERE", "WATER"],
        ["WHERE", "YOU"],
        ["WHERE", "I"],
        ["WHERE", "EAT"],
        ["WHERE", "YOU", "EAT"],
        ["WHERE", "HELP"],
        ["WHERE", "YOU", "WANT", "HOME"],
        ["WHERE", "YOU", "WANT", "EAT"],
        ["YOU", "WANT", "EAT", "WHAT"],
        ["YOU", "WANT", "WHAT"],
        ["WHAT", "YOU", "EAT"],
        ["I", "NAME", "WHAT"],
        ["I", "WANT", "YOU", "HELP"],
        # NEW: questions mixed with other ideas
        ["HELLO", "WHERE", "YOU", "HOME"],
        ["I", "WANT", "WATER", "WHERE", "YOU", "HOME"],
        ["WHERE", "YOU", "HOME", "I", "WANT", "HELP"],
    ]
    for t in tests:
        print(" ".join(t), "->", rule_based(t))