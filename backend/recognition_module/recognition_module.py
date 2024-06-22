import shutil
import csv
import random
import os
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from easyocr.easyocr import Reader
from recognition_module.train_trocr import train_trocr
from easyocr.trainer.train import train

from recognition_module.change_config import update_yaml_parameters, get_config
from recognition_module.recognize_function import *

from handwriting_recognition_service.settings import BASE_DIR
from django.apps import apps


class RecognitionModule:
    USE_GPU = True
    EASYOCR_PATH = os.path.join(BASE_DIR, "recognition_module", "models")
    DATASETS_PATH = os.path.join(BASE_DIR, "recognition_module", "datasets")
    TRAINS_PATH = os.path.join(BASE_DIR, "recognition_module", "trains")  
    CURRENT_MODEL: Model = None
    DETECT_MODEL = Reader(['ru'], gpu=USE_GPU, recognizer=None)

    @staticmethod
    def update_current_model(model_type, model_name):
        if model_type == 0:
            RecognitionModule.CURRENT_MODEL = EasyOCRModel(RecognitionModule.EASYOCR_PATH, model_name, RecognitionModule.USE_GPU)
        elif model_type == 1:      
            RecognitionModule.CURRENT_MODEL = TrOCRModel(os.path.join(RecognitionModule.EASYOCR_PATH, 'model'), model_name, gpu=RecognitionModule.USE_GPU)

    @staticmethod
    def __convert_boxes_to_points(boxes):
        formatted_boxes = []
        for box in boxes:
            x_min, x_max, y_min, y_max = [max(0, coord) for coord in box]
            formatted_box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            formatted_boxes.append(formatted_box)
        return formatted_boxes

    @staticmethod
    def __group_boxes_into_lines(boxes):
        def get_x_coordinates(box):
            x_coords = [point[0] for point in box]
            min_x = min(x_coords)
            max_x = max(x_coords)
            mean_x = sum(x_coords) / len(x_coords)
            return min_x, max_x, mean_x

        # Группировка боксов в строки
        points = np.array([get_x_coordinates(box) for box in boxes])
        min_x_points, max_x_points, mean_x_points = points[:, 0], points[:, 1], points[:, 2]

        cond = mean_x_points[:-1] > min_x_points[1:]
        cond = np.hstack([[True], cond])

        grouped_lines = []
        current_line = []

        for box, is_new_line in zip(boxes, cond):
            if is_new_line:
                if current_line:
                    grouped_lines.append(current_line)
                current_line = [box]
            else:
                current_line.append(box)

        if current_line:
            grouped_lines.append(current_line)

        combined_polygons = []

        for line in grouped_lines:
            if not line:
                continue

            points = []
            for box in line:
                points.extend(box)

            points = np.array(points)
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            combined_polygons.append(hull_points.tolist())

        return grouped_lines, combined_polygons

    @staticmethod
    def __crop_polygon(image_np, polygon):
        """
        Вырезает часть изображения по полигону с белым фоном.
        Parameters:
        image_np (numpy.ndarray): Входное изображение в формате numpy array.
        polygon (list): Полигон, представленный списком точек [[x1, y1], [x2, y2], ...].
        Returns:
        numpy.ndarray: Вырезанная часть изображения.
        tuple: Bounding box координаты (x_min, y_min, x_max, y_max).
        """
        # Преобразуем координаты полигона в целые числа
        polygon = [(int(x), int(y)) for x, y in polygon]

        # Создаем маску
        mask = Image.new('L', (image_np.shape[1], image_np.shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, outline=1, fill=1)
        mask = np.array(mask)

        # Создаем белый фон
        white_background = np.ones_like(image_np) * 255

        # Применяем маску к изображению
        masked_image_np = np.where(mask[..., None], image_np, white_background)

        # Обрезаем по минимальной и максимальной границам полигона
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)
        bbox = (x_min, y_min, x_max, y_max)
        cropped_image_np = masked_image_np[y_min:y_max, x_min:x_max]

        return cropped_image_np, bbox

    @staticmethod
    def __detect_image_craft(image, return_only_lines=True):

        # Определение текста с ограничивающими рамками
        boxes, _ = RecognitionModule.DETECT_MODEL.detect(image, slope_ths=1., reformat = False)

        # Преобразование боксов в нужный формат
        formatted_boxes = RecognitionModule.__convert_boxes_to_points(boxes[0])

        # Группируем боксы в строки и создаем полигоны
        grouped_lines, combined_polygons = RecognitionModule.__group_boxes_into_lines(formatted_boxes)

        if return_only_lines:
            return combined_polygons
        else:
            return formatted_boxes, grouped_lines, combined_polygons

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
    def __extract_text_from_image(image_or_path, model):

        # Проверяем, является ли входное изображение путем или numpy.ndarray
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError("Невозможно загрузить изображение по указанному пути.")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray")

        polygons = RecognitionModule.__detect_image_craft(image=image, return_only_lines=True)

        # Вырезаем bounding boxes из изображения
        cropped_images, boxes = zip(*tuple([RecognitionModule.__crop_polygon(image, polygon) for polygon in polygons]))

        # Распознаем текст из вырезанных изображений
        recognized_texts = model.recognize(image_pieces=cropped_images)

        # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
        results = zip(boxes, recognized_texts)

        return results
    
    @staticmethod
    def __extract_text_from_line(image_or_path, box, model):
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
        
        recognized_text = model.recognize(image_pieces=(cropped_image,))

        return recognized_text[0]

    @staticmethod
    def get_lines_and_text(image):
        RecognitionModule.__check_model()
        if RecognitionModule.CURRENT_MODEL: 
            output = RecognitionModule.__extract_text_from_image(image, RecognitionModule.CURRENT_MODEL)
        else:
            output = []
        output = [(coords, correct_text(line)) for coords, line in output]
        return output
    
    @staticmethod
    def get_text_from_line(image, coords):
        RecognitionModule.__check_model()
        if RecognitionModule.CURRENT_MODEL:
            output = RecognitionModule.__extract_text_from_line(image, coords, RecognitionModule.CURRENT_MODEL)
        else:
            output = ''
        output = correct_text(output)
        return output        

    @staticmethod
    def __check_model():
        AIModel = apps.get_model('ai_service', 'AIModel')
        model = AIModel.objects.get(is_current=True)

        if RecognitionModule.CURRENT_MODEL:
            if not model.model_type == RecognitionModule.CURRENT_MODEL.type or not model.name == RecognitionModule.CURRENT_MODEL.name:
                RecognitionModule.update_current_model(model.model_type, model.name)
        else:
            RecognitionModule.update_current_model(model.model_type, model.name)


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
    