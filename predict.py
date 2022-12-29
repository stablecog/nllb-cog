from typing import List

import torch
from cog import BasePredictor, Input, Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.nllb.constants import TRANSLATOR_MODEL_CACHE, TRANSLATOR_TOKENIZER_CACHE, TRANSLATOR_MODEL_ID
from models.nllb.translate import translate_text

from lingua import LanguageDetectorBuilder


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading language detector...")
        self.detect_language = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        print("Loaded language detector!")
        
        print("Loading translator...")
        self.translate_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_TOKENIZER_CACHE)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATOR_MODEL_ID,
            cache_dir=TRANSLATOR_MODEL_CACHE
        )
        print("Loaded translator!")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        texts: List[str] = Input(description="Input text.", default=[""]),
        text_langs: List[str] = Input(description="Input text language code (FLORES-200). It overrides the language auto-detection.", default=None),
        target_langs: List[str] = Input(description="Target language code (FLORES-200).", default=None),
        target_lang_max_scores: List[float] = Input(
            description="Target language max score for language auto-detection. If detected score is higher than this value, it would override the guess to target_lang as opposed to using detected_lang.",
            default=None
        ),
        label: List[str] = Input(description="A label for the logs.", default=None),
    ) -> List[str]:
        
        default_text_lang=None
        default_target_text_lang = "eng-Latn"
        default_target_lang_max_score = 0.9
        
        if text_langs is None:
            text_langs = [default_text_lang] * len(texts)
        if target_langs is None:
            target_langs = [default_target_text_lang] * len(texts)
        if target_lang_max_scores is None:
            target_langs = [default_target_lang_max_score] * len(texts)
       
        if len(texts) != len(text_langs) or len(texts) != len(target_langs) or len(texts) != len(target_lang_max_scores):
            print("ERROR: texts, text_langs, target_langs, and target_lang_max_scores must be the same length!")
            raise ValueError(
                "Texts, text_langs, target_langs, and target_lang_max_scores must be the same length!"
            )

        output_paths = []
        for i in range(len(texts)):
            text = texts[i]
            text_lang = text_langs[i]
            target_lang = target_langs[i]
            target_lang_max_score = target_lang_max_scores[i]
            translated_text = translate_text(
                text,
                text_lang,
                target_lang,
                target_lang_max_score,
                self.translate_model,
                self.translate_tokenizer,
                self.detect_language,
                label
            )
            output_paths.append(translated_text)
        
        return output_paths