import shutil
import csv
import random
import os
import cv2
import numpy as np
from recognition_module.train_trocr import train_trocr
from easyocr.trainer.train import train

from recognition_module.change_config import update_yaml_parameters, get_config
from recognition_module.recognizer import *
from recognition_module.detector import *
from recognition_module.corrector import *

from handwriting_recognition_service.settings import BASE_DIR
from django.apps import apps


class RecognitionModule:
    USE_GPU = False

    EASYOCR_PATH = os.path.join(BASE_DIR, "recognition_module", "models")
    DATASETS_PATH = os.path.join(BASE_DIR, "recognition_module", "datasets")
    TRAINS_PATH = os.path.join(BASE_DIR, "recognition_module", "trains")  

    DETECTOR: Detector | None = None
    RECOGNIZER: Recognizer | None = None
    CORRECTOR: Corrector | None = None   

    @staticmethod
    def __get_current_detector(detector: int) -> Detector | None:
        AIModel = apps.get_model('ai_service', 'AIModel')
        model = AIModel.objects.get(is_current=True)

        if detector == 0:
            return None
        elif detector == 1:
            if not RecognitionModule.DETECTOR or not isinstance(RecognitionModule.DETECTOR, EasyDetector):
                RecognitionModule.DETECTOR = EasyDetector(detector_network="craft", gpu=RecognitionModule.USE_GPU)
            return RecognitionModule.DETECTOR
        else:
            return None

    @staticmethod
    def __get_current_recognizer(recognizer_name: str, recognizer_type: int) -> Recognizer:
        if RecognitionModule.RECOGNIZER and RecognitionModule.RECOGNIZER.name == recognizer_name and RecognitionModule.RECOGNIZER.type == recognizer_type:
            return RecognitionModule.RECOGNIZER
        else:
            if recognizer_type == 0:
                RecognitionModule.RECOGNIZER = EasyOCRRecognizer(RecognitionModule.EASYOCR_PATH, recognizer_name, RecognitionModule.USE_GPU)
                return RecognitionModule.RECOGNIZER
            elif recognizer_type == 1:
                RecognitionModule.RECOGNIZER = TrOCRRecognizer(os.path.join(RecognitionModule.EASYOCR_PATH, 'model'), recognizer_name, gpu=RecognitionModule.USE_GPU)
                return RecognitionModule.RECOGNIZER
            elif recognizer_type == 3:
                RecognitionModule.RECOGNIZER = DeepTextRecognizer(RecognitionModule.EASYOCR_PATH, recognizer_name, gpu=RecognitionModule.USE_GPU)
                return RecognitionModule.RECOGNIZER
            else:
                return RehandRecognozer()
            
    @staticmethod
    def __get_current_corrector(corrector: int) -> Corrector | None:      
        if corrector == 0:
            return None
        elif corrector == 1:
            if not RecognitionModule.CORRECTOR or not isinstance(RecognitionModule.CORRECTOR, SageCorrector):
                RecognitionModule.CORRECTOR = SageCorrector(gpu=RecognitionModule.USE_GPU)
            return RecognitionModule.CORRECTOR
        elif corrector == 2:
            return YandexTranslateCorrector()
        else:
            return None
        
    @staticmethod
    def __get_setup() -> tuple[Detector | None, Recognizer, Corrector | None]:
        AIModel = apps.get_model('ai_service', 'AIModel')
        model = AIModel.objects.get(is_current=True)

        detector = RecognitionModule.__get_current_detector(model.detector)
        recognizer = RecognitionModule.__get_current_recognizer(model.name, model.model_type)
        corrector = RecognitionModule.__get_current_corrector(model.corrector)

        return detector, recognizer, corrector

    @staticmethod
    def __get_image(image_or_path: str | np.ndarray) -> np.ndarray:
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")
        
        return image

    @staticmethod
    def __extract_text_from_image(image: np.ndarray, recognizer: Recognizer, corrector: Corrector | None, detector: Detector | None=None) -> tuple[tuple[int] | tuple, tuple[str] | tuple]:
        if detector:
            cropped_images, boxes = detector.detect(image)
        else:
            cropped_images, boxes = (image, ), ()

        if isinstance(recognizer, RehandRecognozer) or isinstance(recognizer, DeepTextRecognizer):
            boxes, recognized_texts = recognizer.recognize_and_detect(image)
        else:
            recognized_texts = recognizer.recognize(image_pieces=cropped_images)
        
        if corrector:
            corrected_texts = corrector.batch_correct(recognized_texts)
        else:
            corrected_texts = recognized_texts

        return boxes, corrected_texts

    @staticmethod
    def get_lines_and_text(image_or_path: str | np.ndarray) -> tuple[tuple[tuple[int]] | tuple, tuple[str] | tuple]:
        image = RecognitionModule.__get_image(image_or_path)

        detector, recognizer, corrector = RecognitionModule.__get_setup()
        
        boxes, corrected_texts = RecognitionModule.__extract_text_from_image(image, recognizer, corrector, detector)

        return boxes, corrected_texts
    
    @staticmethod
    def get_text_from_line(image_or_path: str | np.ndarray, coords: list[int] | tuple[int] | np.ndarray) -> str | None:
        image = RecognitionModule.__get_image(image_or_path)

        _, recognizer, corrector = RecognitionModule.__get_setup()

        _, corrected_texts = RecognitionModule.__extract_text_from_image(image, recognizer, corrector)

        if corrected_texts and len(corrected_texts) > 0:
            return corrected_texts[-1]
        else:
            return None      

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
    def __save_image_and_create_csv(data, target_dir):
        if RecognitionModule.DETECTOR:
            images, texts = [], []
            for path_to_image, box, text in data:
                if not text: continue

                filename = os.path.basename(path_to_image).split(".")
                image_name = f"{filename[0]}-{box[0]}-{box[1]}-{box[2]}-{box[3]}.{filename[-1]}"
                target_path = os.path.join(target_dir, image_name)

                image = cv2.imread(path_to_image)
                if image is None:
                    raise ValueError("Невозможно загрузить изображение по указанному пути.")

                cropped_image = RecognitionModule.DETECTOR.__crop_box(image, box)

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
    