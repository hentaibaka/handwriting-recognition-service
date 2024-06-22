from abc import abstractmethod, ABCMeta
import array
import numpy as np
from easyocr.easyocr import Reader
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


class Detector():
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, *args, **kwargs) -> tuple[tuple[np.ndarray], tuple[int]]:
        pass
    
    def crop_polygon(self, image: np.ndarray, polygon):
        polygon = [(int(x), int(y)) for x, y in polygon]

        mask = Image.new('L', (image.shape[1], image.shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, outline=1, fill=1)
        mask = np.array(mask)

        white_background = np.ones_like(image) * 255

        masked_image_np = np.where(mask[..., None], image, white_background)

        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)

        bbox = (x_min, y_min, x_max, y_max)

        cropped_image_np = masked_image_np[y_min:y_max, x_min:x_max]

        return cropped_image_np, bbox
    
    def crop_box(self, image: np.ndarray, box: dict | np.ndarray | list):
        if isinstance(box, dict):
            xmin, ymin, xmax, ymax = box['bbox']
        else:
            xmin, ymin, xmax, ymax = box

        cropped_img = image[ymin:ymax, xmin:xmax]

        return cropped_img

class EasyDetector(Detector):
    def __init__(self, detector_network="craft", gpu=False, *args, **kwargs) -> None:
        self.detector = Reader(['ru'], gpu=gpu, recognizer=False)

    def __convert_boxes_to_points(self, boxes: np.ndarray | list) -> list:
        formatted_boxes = []

        for box in boxes:
            x_min, x_max, y_min, y_max = [max(0, coord) for coord in box]

            formatted_box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

            formatted_boxes.append(formatted_box)

        return formatted_boxes
    
    def __get_x_coordinates(self, box: np.ndarray | list) -> tuple:
        x_coords = [point[0] for point in box]

        min_x = min(x_coords)
        max_x = max(x_coords)

        mean_x = sum(x_coords) / len(x_coords)

        return min_x, max_x, mean_x

    def __group_boxes_into_lines(self, boxes: np.ndarray | list) -> list:
        points = np.array([self.__get_x_coordinates(box) for box in boxes])
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

        return combined_polygons

    def detect(self, image: np.ndarray, *args, **kwargs) -> tuple[tuple[np.ndarray], tuple[int]]:
        boxes, _ = self.detector.detect(image, slope_ths=1., reformat = False)

        formatted_boxes = self.__convert_boxes_to_points(boxes[0])

        combined_polygons = self.__group_boxes_into_lines(formatted_boxes)

        cropped_images, bboxes = zip(*tuple([self.crop_polygon(image, polygon) for polygon in combined_polygons]))

        return cropped_images, bboxes