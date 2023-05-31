from .constants import (
    TARGET_LANG_FLORES,
    TARGET_LANG_SCORE_MAX,
    DETECTED_CONFIDENCE_SCORE_MIN,
)


class TranslationInput:
    def __init__(
        self,
        text: str = "",
        text_flores: str = None,
        target_flores: str = TARGET_LANG_FLORES,
        target_score_max: float = TARGET_LANG_SCORE_MAX,
        detected_confidence_score_min: float = DETECTED_CONFIDENCE_SCORE_MIN,
    ):
        self.text = text
        self.text_flores = text_flores
        self.target_flores = target_flores
        self.target_score_max = target_score_max
        self.detected_confidence_score_min = detected_confidence_score_min


class TranslationOutput:
    def __init__(
        self,
        original_text: str,
        original_text_guessed_flores: str,
        translated_text: str,
        translated_text_flores: str,
    ):
        self.original_text = original_text
        self.original_text_guessed_flores = original_text_guessed_flores
        self.translated_text = translated_text
        self.translated_text_flores = translated_text_flores
