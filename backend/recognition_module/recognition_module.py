from logging import raiseExceptions
import shutil
import csv
import random
import os
import cv2
import numpy as np
import torch.backends.cudnn as cudnn

from recognition_module.train_trocr import train_trocr
from easyocr.trainer.train import train

from recognition_module.change_config import update_yaml_parameters, get_config
from yolo9.detect_function import detect_image
from recognition_module.recognize_function import recognize_text_from_images, recognize_text_from_image, recognize_text_from_imagesTrOCR

from handwriting_recognition_service.settings import BASE_DIR
from django.apps import apps


class RecognitionModule:
    YOLO_MODEL = os.path.join(BASE_DIR, "recognition_module", "models", "yolo_line_detection.pt")
    EASYOCR_PATH = os.path.join(BASE_DIR, "recognition_module", "models")
    DATASETS_PATH = os.path.join(BASE_DIR, "recognition_module", "datasets")
    TRAINS_PATH = os.path.join(BASE_DIR, "recognition_module", "trains")        

    @staticmethod
    def __save_image_and_create_csv(data, target_dir):
        images, texts = [], []
        for path_to_image, box, text in data:
            if not text: continue

            filename = os.path.basename(path_to_image).split(".")
            image_name = f"{filename[0]}-{box[0]}-{box[1]}-{box[2]}-{box[3]}.{filename[-1]}"
            target_path = os.path.join(target_dir, image_name)

            image = cv2.imread(path_to_image)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
            
            cropped_image = RecognitionModule.__crop_box_from_image(image, box)
            
            cv2.imwrite(target_path, cropped_image)

            images.append(image_name)
            texts.append(text)
        
        # Создание labels.csv
        labels_file = os.path.join(target_dir, 'labels.csv')
        with open(labels_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'words'])
            content = list(zip(images, texts))
            if content:
                writer.writerows(content[:-1])
                writer = csv.writer(file, lineterminator="")
                writer.writerow(content[-1])

    @staticmethod
    def __crop_box_from_image(image, box):
        if isinstance(image, np.ndarray):
            if isinstance(box, dict):
                xmin, ymin, xmax, ymax = box['bbox']
            else:
                xmin, ymin, xmax, ymax = box
            cropped_img = image[ymin:ymax, xmin:xmax]
            return cropped_img
        else:
            raise ValueError("image должен быть объектом numpy.ndarray")    
    
    @staticmethod
    def __extract_text_from_image(image_or_path, yolo_weights_path, ocr_models_directory, recog_network, yolo_device="cpu", ocr_gpu=False):
        """
        Extracts text from an image using YOLO for object detection and EasyOCR for text recognition.

        Parameters:
        image_or_path (str or np.ndarray): Path to the image or the image itself as a numpy.ndarray.
        yolo_weights_path (str): Path to the YOLO weights file.
        ocr_models_directory (str): Directory containing OCR models.
                                    Example structure:
                                    ocr_models_directory/model/best_accuracy.pth
                                    ocr_models_directory/user_network/best_accuracy.py
                                    ocr_models_directory/user_network/best_accuracy.yaml
        recog_network (str): Name of the OCR network to use (default is "best_accuracy").
        yolo_device (str): Device to run YOLO on, either "cpu" or the GPU index (default is "cpu").
        ocr_gpu (bool): Whether to use GPU for OCR (default is False).

        Returns:
        list: A list of tuples where each tuple contains bounding box coordinates and the recognized text.
              Format: [([x1, y1, x2, y2], "string"), ...]
        """

        # Проверяем, является ли входное изображение путем или numpy.ndarray
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")

        # Обнаруживаем bounding boxes
        boxes = detect_image(weights=yolo_weights_path, source=image, sort_boxes=True, device=yolo_device)

        # Вырезаем bounding boxes из изображения
        cropped_images = [RecognitionModule.__crop_box_from_image(image, box) for box in boxes]

        # Распознаем текст из вырезанных изображений
        recognized_texts = recognize_text_from_images(image_pieces=cropped_images, models_directory=ocr_models_directory, recog_network=recog_network, gpu=ocr_gpu)

        # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
        results = [([box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]], text) for box, text in zip(boxes, recognized_texts)]

        return results
    
    @staticmethod
    def __extract_text_from_line(image_or_path, box, ocr_models_directory, recog_network, ocr_gpu=False):
        # Проверяем, является ли входное изображение путем или numpy.ndarray
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")
        
        cropped_image = RecognitionModule.__crop_box_from_image(image, box)
        
        recognized_text = recognize_text_from_image(cropped_image, models_directory=ocr_models_directory, recog_network=recog_network, gpu=ocr_gpu)

        return recognized_text

    @staticmethod
    def __extract_text_from_imageTrOCR(image_or_path, yolo_weights_path, TrOCR_directory, recog_network, yolo_device="cpu", ocr_gpu=False):
        """
        Extracts text from an image using YOLO for object detection and EasyOCR for text recognition.

        Parameters:
        image_or_path (str or np.ndarray): Path to the image or the image itself as a numpy.ndarray.
        yolo_weights_path (str): Path to the YOLO weights file.
        ocr_models_directory (str): Directory containing OCR models.
                                    Example structure:
                                    ocr_models_directory/model/best_accuracy.pth
                                    ocr_models_directory/user_network/best_accuracy.py
                                    ocr_models_directory/user_network/best_accuracy.yaml
        recog_network (str): Name of the OCR network to use (default is "best_accuracy").
        yolo_device (str): Device to run YOLO on, either "cpu" or the GPU index (default is "cpu").
        ocr_gpu (bool): Whether to use GPU for OCR (default is False).

        Returns:
        list: A list of tuples where each tuple contains bounding box coordinates and the recognized text.
              Format: [([x1, y1, x2, y2], "string"), ...]
        """

        # Проверяем, является ли входное изображение путем или numpy.ndarray
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")

        # Обнаруживаем bounding boxes
        boxes = detect_image(weights=yolo_weights_path, source=image, sort_boxes=True, device=yolo_device)

        # Вырезаем bounding boxes из изображения
        cropped_images = [RecognitionModule.__crop_box_from_image(image, box) for box in boxes]

        TrOCR_directory = os.path.join(TrOCR_directory, 'model', recog_network)

        # Распознаем текст из вырезанных изображений
        recognized_texts = recognize_text_from_imagesTrOCR(image_pieces=cropped_images, models_directory=TrOCR_directory, gpu=ocr_gpu)

        # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
        results = [([box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]], text) for box, text in zip(boxes, recognized_texts)]

        return results

    @staticmethod
    def __extract_text_from_lineTrOCR(image_or_path, box, TrOCR_directory, recog_network, ocr_gpu=False):
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")
    
        cropped_image = RecognitionModule.__crop_box_from_image(image, box)

        TrOCR_directory = os.path.join(TrOCR_directory, 'model', recog_network)

        # Распознаем текст из вырезанных изображений
        recognized_texts = recognize_text_from_imagesTrOCR(image_pieces=[cropped_image, ], models_directory=TrOCR_directory, gpu=ocr_gpu)

        # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
        results = recognized_texts[0]

        return results

    @staticmethod
    def get_lines_and_text(image):
        AIModel = apps.get_model('ai_service', 'AIModel')
        model = AIModel.objects.get(is_current=True)
        if model and model.model_type == 0: # type: ignore
            output = RecognitionModule.__extract_text_from_image(image, 
                                                                RecognitionModule.YOLO_MODEL,
                                                                RecognitionModule.EASYOCR_PATH,
                                                                recog_network=model.name) # type: ignore
        elif model and model.model_type == 1: # type: ignore
            output = RecognitionModule.__extract_text_from_imageTrOCR(image,
                                                                      RecognitionModule.YOLO_MODEL,
                                                                      RecognitionModule.EASYOCR_PATH,
                                                                      recog_network=model.name) # type: ignore
        else:
            result = []

        return output
    
    @staticmethod
    def get_text_from_line(image, coords):
        AIModel = apps.get_model('ai_service', 'AIModel')
        model = AIModel.objects.get(is_current=True)
        
        if model and model.model_type == 0: # type: ignore
            output = RecognitionModule.__extract_text_from_line(image, 
                                                                coords,
                                                                RecognitionModule.EASYOCR_PATH,
                                                                recog_network=model.name) # type: ignore
        elif model and model.model_type == 1: # type: ignore
            output = RecognitionModule.__extract_text_from_lineTrOCR(image, 
                                                                     coords,
                                                                     RecognitionModule.EASYOCR_PATH,
                                                                     recog_network=model.name) # type: ignore
        else:
            result = []
        return output        

    @staticmethod
    def train(data, dataset_name, model_name_train, model_type, model_name_trained, num_iter, val_interval, batch_size, new_prediction, train_perc=0.9):
        # Получаем полный путь до датасета
        dataset_path = os.path.join(RecognitionModule.DATASETS_PATH, dataset_name)
        # Создание основной папки датасета
        os.makedirs(dataset_path, exist_ok=True)

        result = (True, "Обучение прошло успешно")
        try:
            if not data: raise Exception("Набор данных не должен быть пустым")
            # Создание папок train и val внутри основной папки
            train_dir = os.path.join(dataset_path, 'train')
            val_dir = os.path.join(dataset_path, 'val')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
    
            # Разделение данных на train и val (train_perc% и 100-train_perc%)
            random.shuffle(data)
            split_idx = int(train_perc * len(data))
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            if not train_data: raise Exception("Тренировочный набор не должен быть пустым, попробуйте добавить больше данных в датасет")
            if not val_data: raise Exception("Валидационный набор не должен быть пустым, попробуйте добавить больше данных в датасет")
    
            # Копирование файлов и создание labels.csv для train и val
            RecognitionModule.__save_image_and_create_csv(train_data, train_dir)
            RecognitionModule.__save_image_and_create_csv(val_data, val_dir)
        
            train_path = os.path.join(RecognitionModule.TRAINS_PATH, model_name_trained)

            os.makedirs(train_path, exist_ok=True)

            if model_type == 0:
                trained_model_path = os.path.join(RecognitionModule.EASYOCR_PATH, 'model', model_name_trained + '.pth')

                config_file = os.path.join(RecognitionModule.EASYOCR_PATH, 'user_network', model_name_train + '.yaml')

                output_file = RecognitionModule.generate_config(config_file, dataset_name, 'train', val_dir, model_name_train, model_name_trained,
                                                                num_iter=num_iter, valInterval=val_interval, batch_size=batch_size, new_prediction=new_prediction)
                opt = get_config(output_file)
                train(opt, train_path, trained_model_path, true_if_acc_else_norm_ED=True)

                shutil.copy(output_file, 
                            os.path.join(RecognitionModule.EASYOCR_PATH, 'user_network', model_name_trained + '.yaml'))
                shutil.copy(os.path.join(RecognitionModule.EASYOCR_PATH, 'user_network', model_name_train + '.py'),
                            os.path.join(RecognitionModule.EASYOCR_PATH, 'user_network', model_name_trained + '.py'))
            elif model_type == 1:
                train_trocr(train_csv=os.path.join(train_dir, 'labels.csv'),
                            train_root=train_dir,
                            val_csv=os.path.join(val_dir, 'labels.csv'),
                            val_root=val_dir,
                            model_name=os.path.join(RecognitionModule.EASYOCR_PATH, 'model', model_name_train),
                            output_dir=os.path.join(RecognitionModule.EASYOCR_PATH, 'model', model_name_trained),
                            epochs=num_iter,
                            batch_size=batch_size,
                            learning_rate=5e-5,
                            gpu=False)
            else:
                raise Exception("Неподдерживаемый тип модели")


        except Exception as ex:
            #raise ex
            result = (False, ex)

        shutil.rmtree(dataset_path)
        
        return result

    @staticmethod
    def generate_config(config_file, dataset_name, train_folder, val_dir, model_name_train, model_name_trained,
                        num_iter=100, valInterval=50, new_prediction=True, batch_size=64):
        dataset_path = os.path.join(RecognitionModule.DATASETS_PATH, dataset_name)

        updates = {
            'experiment_name': model_name_trained,
            'num_iter': num_iter,
            'valInterval': valInterval,
            'train_data': dataset_path,
            'valid_data': val_dir,
            'select_data': train_folder,
            'saved_model': os.path.join(RecognitionModule.EASYOCR_PATH, 'model', model_name_train + '.pth'),
            'path_save_model': RecognitionModule.DATASETS_PATH,
            'new_prediction': new_prediction,
            'batch_size': batch_size,
        }

        output_file = os.path.join(dataset_path, dataset_name + '.yaml')
        
        update_yaml_parameters(config_file, 
                               updates, 
                               output_file)
        
        return output_file
    