from abc import abstractmethod, ABC
import os
import cv2
import numpy as np
from easyocr.easyocr import Reader
from easyocr.trainer.utils import AttnLabelConverter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import requests
import time
from random import randint


class Recognizer(ABC):
    @abstractmethod
    def __init__(self, models_path, recog_network, gpu=False, *args, **kwargs):
        pass

    @abstractmethod
    def recognize(self, image_pieces: tuple[np.ndarray], *args, **kwargs) -> tuple[str] | tuple:
        pass

class RehandRecognozer(Recognizer):
    def __init__(self, *args, **kwargs):
        self.name = 'crocr'
        self.type = 2

    def __upload_photo(self, file_bytes: io.BytesIO) -> dict:
        url = "https://rehand.ru/api/v1/upload"

        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Host": "rehand.ru",
            "Origin": "https://rehand.ru",
            "Referer": "https://rehand.ru/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/52.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\""
        }
        
        files = {
            'file': ('image.jpg', file_bytes, 'image/jpg')
        }
        
        data = {
            "language": "russian",
            "correctOrder": "on",
            "speller": "on",
            "handwritingText": "on"
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.RequestException as e:
            return {"error": "Request failed.", "exception": str(e)}
    
        if response.status_code == 200 and response.text:
            try:
                result = response.json()
                return result
            except ValueError:
                return {"error": "Response is not JSON."}
        else:
            return {"error": "Request failed.", "status_code": response.status_code, "response_text": response.text}

    def __convert_boxes_to_points(self, boxes: list) -> list:
        formatted_boxes = []

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            formatted_box = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
            formatted_boxes.append(formatted_box)

        return formatted_boxes
    
    def __get_x_coordinates(self, box: list) -> tuple:
        x_coords = [point[0] for point in box]
        min_x = min(x_coords)
        max_x = max(x_coords)
        mean_x = sum(x_coords) / len(x_coords)
        return min_x, max_x, mean_x

    def __group_boxes_into_lines(self, boxes: list, words: list) -> list:
        points = np.array([self.__get_x_coordinates(box) for box in boxes])
        min_x_points, max_x_points, mean_x_points = points[:, 0], points[:, 1], points[:, 2]
        cond = mean_x_points[:-1] > min_x_points[1:]
        cond = np.hstack([[True], cond])

        grouped_lines = []
        current_line_boxes = []
        current_line_text = []

        for box, is_new_line, word in zip(boxes, cond, words):
            if is_new_line:
                if current_line_boxes and current_line_text:
                    grouped_lines.append((current_line_boxes, ' '.join(current_line_text)))
                current_line_boxes = [box]
                current_line_text = [word]
            else:
                current_line_boxes.append(box)
                current_line_text.append(word)

        if current_line_boxes and current_line_text:
            grouped_lines.append((current_line_boxes, ' '.join(current_line_text)))

        return grouped_lines

    def recognize(self, image_pieces: tuple[np.ndarray], *args, **kwargs) -> tuple[str] | tuple:
        results = []
        for image_piece in image_pieces:
            is_success, buffer = cv2.imencode(".jpg", image_piece)
            if not is_success: continue

            file_bytes = io.BytesIO(buffer)

            response = self.__upload_photo(file_bytes)
            if "error" in response: continue

            results.append(response.get('output_text', '').replace('<br>', ' '))

        return tuple(results)

    def recognize_and_detect(self, image: np.ndarray) -> tuple[tuple[tuple[int]] | tuple, tuple[str] | tuple]:
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success: 
            return (), ()
        
        file_bytes = io.BytesIO(buffer)
        response = self.__upload_photo(file_bytes)

        if "error" in response: 
            return (), ()

        boxes = []
        words = []
        for box in response.get("boxes", []):
            x1 = box["x"]
            y1 = box["y"]
            x2 = x1 + box["w"]
            y2 = y1 + box["h"]
            word = box["text"]

            boxes.append([x1, y1, x2, y2])
            words.append(word)

        formatted_boxes = self.__convert_boxes_to_points(boxes)
        grouped_lines = self.__group_boxes_into_lines(formatted_boxes, words)
        coords = []
        texts = []
        for boxes, text in grouped_lines:
            c1 = boxes[0][0]
            c2 = boxes[-1][2]
            coords.append(c1 + c2)
            texts.append(text)

        time.sleep(randint(5, 10))

        return tuple(coords), tuple(texts)

class EasyOCRRecognizer(Recognizer):
    def __init__(self, models_path, recog_network, gpu=False, *args, **kwargs):
        self.name = recog_network
        self.type = 0
        self.model_storage_directory = os.path.join(models_path, "model")
        self.user_network_directory = os.path.join(models_path, "user_network")

        self.reader = Reader(['ru'], recog_network=recog_network, gpu=gpu,
                            model_storage_directory=self.model_storage_directory,
                            user_network_directory=self.user_network_directory)
        
    def recognize(self, image_pieces: tuple[np.ndarray], *args, **kwargs) -> tuple[str] | tuple:
        recognized_texts = []

        for image_piece in image_pieces:
            image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)

            if hasattr(self.reader, 'converter') and isinstance(self.reader.converter, AttnLabelConverter):
                result = self.reader.readtext(image_cv, detail=0, decoder="beamsearch")
            else:
                result = self.reader.readtext(image_cv, detail=0)

            recognized_texts.append(" ".join(result))
        return tuple(recognized_texts)

class TrOCRRecognizer(Recognizer):
    def __init__(self, models_path, recog_network, gpu=False, *args, **kwargs):
        self.name = recog_network
        self.type = 1
        self.device = torch.device('cuda' if gpu else 'cpu')
        models_directory = os.path.join(models_path, recog_network)
        self.processor = TrOCRProcessor.from_pretrained(models_directory)
        self.model = VisionEncoderDecoderModel.from_pretrained(models_directory)
        self.model.to(self.device)


    def recognize(self, image_pieces: tuple[np.ndarray], batch_size=8, *args, **kwargs) -> tuple[str] | tuple:
        recognized_texts = []

        for i in range(0, len(image_pieces), batch_size):
            batch_images = image_pieces[i:i + batch_size]

            batch_cv_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in batch_images]

            pixel_values = self.processor(images=batch_cv_images, return_tensors="pt").pixel_values.to(self.device)

            generated_ids = self.model.generate(pixel_values)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            recognized_texts.extend(generated_texts)
        return tuple(recognized_texts)
