from typing import List

import torch
from cog import BasePredictor, Input

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.nllb.constants import (
    TRANSLATOR_MODEL_CACHE,
    TRANSLATOR_TOKENIZER_CACHE,
    TRANSLATOR_MODEL_ID,
    TARGET_LANG,
    TARGET_LANG_SCORE_MAX,
    DETECTED_CONFIDENCE_SCORE_MIN,
)
from models.nllb.translate import translate_text

from lingua import LanguageDetectorBuilder


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading language detector...")
        self.detect_language = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
        print("Loaded language detector!")

        print("Loading translator...")
        self.translate_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_TOKENIZER_CACHE
        )
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_MODEL_CACHE
        )
        print("Loaded translator!")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        text: str = Input(description="Input text.", default=""),
        text_lang: str = Input(
            description="Input text language code (FLORES-200). It overrides the language auto-detection.",
            default=None,
        ),
        target_lang: str = Input(
            description="Target language code (FLORES-200).", default=TARGET_LANG
        ),
        target_lang_max_score: float = Input(
            description="Target language max score for language auto-detection. If detected score is higher than this value, it would override the guess to target_lang as opposed to using detected_lang.",
            default=TARGET_LANG_SCORE_MAX,
        ),
        detected_confidence_score_min: float = Input(
            description="#1 - Minimum confidence score for language auto-detection. If detected score is lower than this value, it would override the guess to target_lang as opposed to using detected_lang.",
            default=DETECTED_CONFIDENCE_SCORE_MIN,
        ),
        label: str = Input(description="A label for the logs.", default="Text"),
        text_2: str = Input(description="#2 - Input text.", default=None),
        text_lang_2: str = Input(
            description="#2 - Input text language code (FLORES-200). It overrides the language auto-detection.",
            default=None,
        ),
        target_lang_2: str = Input(
            description="#2 - Target language code (FLORES-200).", default=TARGET_LANG
        ),
        target_lang_max_score_2: float = Input(
            description="#2 - Target language max score for language auto-detection. If detected score is higher than this value, it would override the guess to target_lang as opposed to using detected_lang.",
            default=TARGET_LANG_SCORE_MAX,
        ),
        detected_confidence_score_min_2: float = Input(
            description="#2 - Minimum confidence score for language auto-detection. If detected score is lower than this value, it would override the guess to target_lang as opposed to using detected_lang.",
            default=DETECTED_CONFIDENCE_SCORE_MIN,
        ),
        label_2: str = Input(description="#2 - A label for the logs.", default="Text"),
    ) -> List[str]:
        output_strings = []
        translated_text = translate_text(
            text=text,
            text_flores=text_lang,
            target_lang=target_lang,
            target_flores=target_lang,
            detected_confidence_score_min=detected_confidence_score_min,
            target_lang_max_score=target_lang_max_score,
            model=self.translate_model,
            tokenizer=self.translate_tokenizer,
            detector=self.detect_language,
            label=label,
        )
        output_strings.append(translated_text)
        if text_2 is not None:
            translated_text_2 = translate_text(
                text=text_2,
                text_flores=text_lang_2,
                target_lang=target_lang_2,
                target_flores=target_lang_2,
                detected_confidence_score_min=detected_confidence_score_min_2,
                target_lang_max_score=target_lang_max_score_2,
                model=self.translate_model,
                tokenizer=self.translate_tokenizer,
                detector=self.detect_language,
                label=label_2,
            )
            output_strings.append(translated_text_2)
        return output_strings
