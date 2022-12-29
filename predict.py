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
        text: str = Input(description="Input text.", default=""),
        text_lang: str = Input(description="Text language code (FLORES-200). It overrides the language auto-detection.", default=None),
        target_lang: str = Input(description="Target language code (FLORES-200).", default="eng_Latn"),
        target_lang_max_score: float = Input(description="Target language max score.", default=0.9),
        label: str = Input(description="A label for the logs.", default="Text"),
    ) -> List[Path]:
        output_paths = []
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