from abc import abstractmethod
import os
import cv2
import numpy as np
from easyocr.easyocr import Reader
from easyocr.trainer.utils import AttnLabelConverter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection


class Model:
    @abstractmethod
    def __init__(self, models_path, recog_network, gpu) -> None:
        pass

    @abstractmethod
    def recognize(self, image_pieces) -> list:
        pass

class EasyOCRModel(Model):
    def __init__(self, models_path, recog_network, gpu=False):
        self.name=recog_network
        self.type = 0
        self.model_storage_directory = os.path.join(models_path, "model")
        self.user_network_directory = os.path.join(models_path, "user_network")

        self.reader = Reader(['ru'], recog_network=recog_network, gpu=gpu,
                            model_storage_directory=self.model_storage_directory,
                            user_network_directory=self.user_network_directory)
        
    def recognize(self, image_pieces) -> list:
        recognized_texts = []

        for image_piece in image_pieces:
            image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)

            if hasattr(self.reader, 'converter') and isinstance(self.reader.converter, AttnLabelConverter):
                result = self.reader.readtext(image_cv, detail=0, decoder="beamsearch")
            else:
                result = self.reader.readtext(image_cv, detail=0)

            recognized_texts.append(" ".join(result))
        return recognized_texts

class TrOCRModel(Model):
    def __init__(self, models_path, recog_network, gpu=False):
        self.name = recog_network
        self.type = 1
        self.device = torch.device('cuda' if gpu else 'cpu')
        models_directory = os.path.join(models_path, recog_network)
        self.processor = TrOCRProcessor.from_pretrained(models_directory)
        self.model = VisionEncoderDecoderModel.from_pretrained(models_directory)
        self.model.to(self.device)


    def recognize(self, image_pieces, batch_size=8) -> list:
        recognized_texts = []

        for i in range(0, len(image_pieces), batch_size):
            batch_images = image_pieces[i:i + batch_size]

            batch_cv_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in batch_images]

            pixel_values = self.processor(images=batch_cv_images, return_tensors="pt").pixel_values.to(self.device)

            generated_ids = self.model.generate(pixel_values)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            recognized_texts.extend(generated_texts)
        return recognized_texts

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
    device = torch.device('cuda:0' if gpu else 'cpu')

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
