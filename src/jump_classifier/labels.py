CLASS_NAMES = ["Axel", "Lutz", "Flip", "Loop", "Salchow", "Toe Loop"]

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def normalize_label_name(name: str) -> str:
    """Match folder names with small differences such as ToeLoop vs Toe Loop."""
    cleaned = name.strip().lower().replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())

    aliases = {
        "axel": "Axel",
        "lutz": "Lutz",
        "flip": "Flip",
        "loop": "Loop",
        "salchow": "Salchow",
        "toe loop": "Toe Loop",
        "toeloop": "Toe Loop",
    }
    if cleaned not in aliases:
        raise ValueError(
            f"Unknown class folder '{name}'. Expected one of: {', '.join(CLASS_NAMES)}"
        )
    return aliases[cleaned]
