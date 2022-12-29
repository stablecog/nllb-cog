from lingua import Language
import time
from transformers import pipeline
from .constants import LANG_TO_FLORES, FLORES_TO_LANG
import torch


def translate_text(text, text_flores, target_flores, target_score_max, model, tokenizer, detector, label):
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""
    startTimeTranslation = time.time()
    translated_text = ""
    decided_text_flores = target_flores
    
    if text_flores != None:
        decided_text_flores = text_flores
        print(f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{decided_text_flores}" --')
    else:
        confidence_values = detector.compute_language_confidence_values(text)
        target_lang_score = None
        detected_lang = None
        detected_lang_score = None
        
        print(f'-- Confidence values - {confidence_values[:10]} --')
        for index in range(len(confidence_values)):
            curr = confidence_values[index]
            if index == 0:
                detected_lang = curr[0]
                detected_lang_score = curr[1]
            if curr[0] == Language.ENGLISH:
                target_lang_score = curr[1]
                
        if detected_lang is not None and LANG_TO_FLORES.get(detected_lang) != target_flores and (target_lang_score is None or target_lang_score < target_score_max) and LANG_TO_FLORES.get(detected_lang.name) is not None:
            decided_text_flores = LANG_TO_FLORES[detected_lang.name]
        
        if detected_lang is not None:
            print(f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --')
        if detected_lang is not None and target_lang_score is not None and detected_lang != target_flores:
            print(f'-- {label} - Target language score: {target_lang_score} --')
        
        print(f'-- {label} - Selected text language FLORES-200: "{decided_text_flores}" --')

    if decided_text_flores != target_flores:
        translate = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            src_lang=decided_text_flores,
            tgt_lang=target_flores,
            device=0
        )
        translate_output = translate(text, max_length=1000)
        translated_text = translate_output[0]['translation_text']
        print(f'-- {label} - Original text is: "{text}" --')
        print(f'-- {label} - Translated text is: "{translated_text}" --')
    else:
        translated_text = text
        print(f"-- {label} - Text is already in the correct language, no translation needed --")
    
    endTimeTranslation = time.time()
    print(f"-- {label} - Completed in: {round((endTimeTranslation - startTimeTranslation), 2)} sec. --")
    
    return translated_text