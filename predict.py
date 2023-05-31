from typing import List

import torch
from cog import BasePredictor, Input

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.nllb.constants import (
    TRANSLATOR_CACHE,
    TOKENIZER_CACHE,
    TRANSLATOR_MODEL_ID,
)
from models.nllb.classes import TranslationInput, TranslationOutput
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
            TRANSLATOR_MODEL_ID, cache_dir=TOKENIZER_CACHE
        )
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        )
        print("Loaded translator!")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        text_objects: List[TranslationInput] = Input(
            description="Translation inputs", default=[]
        ),
    ) -> List[TranslationOutput]:
        outputs: List[TranslationOutput] = []
        for text_object in text_objects:
            translated_text_object = translate_text(
                text=text_object.original_text,
                text_flores=text_object.text_flores,
                target_flores=text_object.target_flores,
                detected_confidence_score_min=text_object.detected_confidence_score_min,
                target_score_max=text_object.target_score_max,
                model=self.translate_model,
                tokenizer=self.translate_tokenizer,
                detector=self.detect_language,
            )
            outputs.append(translated_text_object)
        return outputs
