"""
Vocabulary for the ASL project. Used by collect_sequences.py, train.py and
predict.py, so change the words HERE only.

Folder / label names use underscores (THANK_YOU); on screen they show as
"THANK YOU".
"""

# One-hand signs: collect these first.
ONE_HAND_SIGNS = [
    "HELLO", "BYE", "THANK_YOU", "PLEASE", "SORRY",
    "I", "YOU", "YES", "NO", "WATER",
    "EAT", "TOILET", "WHERE", "GOOD", "BAD", "HOME",
]

# Two-hand signs: both hands must be visible while signing.
TWO_HAND_SIGNS = ["HELP", "WANT", "WHAT", "NAME"]

# "Not signing": relaxed hand, random movement, or no hand in the frame.
IDLE = "IDLE"

SIGNS = ONE_HAND_SIGNS + TWO_HAND_SIGNS + [IDLE]


def display_name(sign):
    return sign.replace("_", " ")