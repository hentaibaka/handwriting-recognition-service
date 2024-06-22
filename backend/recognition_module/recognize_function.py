import os
import cv2
import numpy as np
from easyocr.easyocr import Reader
from easyocr.trainer.utils import AttnLabelConverter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection

def correct_text(text: str) -> str:
    corrector_m2m = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
    result = corrector_m2m.correct(text)
    return result[0]

def recognize_text_from_images(image_pieces, models_directory, recog_network='best_accuracy', gpu=False):
    model_storage_directory = os.path.join(models_directory, "model")
    user_network_directory = os.path.join(models_directory, "user_network")

    reader = Reader(['ru'], recog_network=recog_network, gpu=gpu,
                            model_storage_directory=model_storage_directory,
                            user_network_directory=user_network_directory)
    
    recognized_texts = []
    for image_piece in image_pieces:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)
        # Perform text recognition
        if hasattr(reader, 'converter') and isinstance(reader.converter, AttnLabelConverter):
            result = reader.readtext(image_cv, detail=0, decoder="beamsearch")
        else:
            result = reader.readtext(image_cv, detail=0)
        recognized_texts.append(" ".join(result))
    
    return recognized_texts

def recognize_text_from_imagesTrOCR(image_pieces, models_directory, gpu=False):
    device = torch.device('cuda' if gpu else 'cpu')

    processor = TrOCRProcessor.from_pretrained(models_directory)
    model = VisionEncoderDecoderModel.from_pretrained(models_directory)
    model.to(device)

    recognized_texts = []
    for image_piece in image_pieces:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)
        # Perform text recognition
        pixel_values = processor(images=image_cv, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values.to(device))
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        recognized_texts.append(generated_text[0])
    
    return recognized_texts
