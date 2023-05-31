from lingua import Language
import time
from transformers import pipeline
from .constants import LANG_TO_FLORES, FLORES_TO_LANG, TARGET_LANG
import torch
from tabulate import tabulate
from .classes import TranslationOutput
from typing import Any, List


def translate_text(
    text: str,
    text_flores: str | None,
    target_flores: str,
    target_score_max: float,
    detected_confidence_score_min: float,
    model: Any,
    tokenizer: Any,
    detector: Any,
) -> TranslationOutput:
    translated_text_object = TranslationOutput(
        original_text=text,
        original_text_guessed_flores=target_flores,
        translated_text=text,
        translated_text_flores=target_flores,
    )
    if text == "":
        print(f"-- No text to translate, skipping --")
        return translated_text_object
    startTimeTranslation = time.time()
    translated_text: str = ""
    decided_text_flores = get_flores(
        text=text,
        text_flores=text_flores,
        target_flores=target_flores,
        target_score_max=target_score_max,
        detected_confidence_score_min=detected_confidence_score_min,
        detector=detector,
    )
    translated_text_object.original_text_guessed_flores = decided_text_flores

    if decided_text_flores != target_flores:
        translate = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            src_lang=decided_text_flores,
            tgt_lang=target_flores,
            device=0,
        )
        translate_output = translate(text, max_length=1000)
        translated_text = translate_output[0]["translation_text"]
        translated_text_object.translated_text = translated_text
        print(f'-- Original text is: "{text}" --')
        print(f'-- Translated text is: "{translated_text}" --')
    else:
        print(f"-- Text is already in the correct language, no translation needed --")

    endTimeTranslation = time.time()
    print(
        f"-- Completed in: {round((endTimeTranslation - startTimeTranslation), 2)} sec. --"
    )
    return translated_text_object


def get_flores(
    text,
    text_flores,
    target_flores,
    target_score_max,
    detected_confidence_score_min,
    detector,
):
    if text == "":
        print(f"-- No text to give FLORES-200 for, skipping --")
        return target_flores
    if text_flores is not None and text_flores != "":
        print(
            f'-- FLORES-200 code is given, skipping language auto-detection: "{text_flores}" --'
        )
        return text_flores

    text_lang_flores = target_flores
    confidence_values = detector.compute_language_confidence_values(text)
    target_lang_score = None
    detected_lang = None
    detected_lang_score = None

    target_lang = TARGET_LANG
    if FLORES_TO_LANG.get(target_flores) is not None:
        target_lang = FLORES_TO_LANG[target_flores]

    print(f"-- Confidence values --")
    print(tabulate(confidence_values[:5]))

    for index in range(len(confidence_values)):
        curr = confidence_values[index]
        if index == 0:
            detected_lang = curr[0]
            detected_lang_score = curr[1]
        if curr[0] == Language.ENGLISH:
            target_lang_score = curr[1]

    if (
        detected_lang is not None
        and detected_lang != target_lang
        and detected_lang_score > detected_confidence_score_min
        and (target_lang_score is None or target_lang_score < target_score_max)
        and LANG_TO_FLORES.get(detected_lang.name) is not None
    ):
        text_lang_flores = LANG_TO_FLORES[detected_lang.name]

    if detected_lang is not None:
        print(
            f'-- Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --'
        )
    if (
        detected_lang is not None
        and target_lang_score is not None
        and detected_lang != target_lang
    ):
        print(f"-- Target language score: {target_lang_score} --")

    print(f'-- Selected text language FLORES-200: "{text_lang_flores}" --')
    return text_lang_flores
