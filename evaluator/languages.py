SUPPORTED_LANGUAGES = ["hi", "ta"]

def is_supported(lang_code: str) -> bool:
    """Check if the given language code is supported."""
    return lang_code in SUPPORTED_LANGUAGES
