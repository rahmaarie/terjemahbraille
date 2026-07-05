from collections import Counter
from pathlib import Path
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

from .convert import parse_xywh_and_class


class BrailleSegmentation:
    def __init__(self, yolo_weight="weights/yolov8_braille.pt"):
        self.conf = 0.15
        self.image_dim = (100, 150)

        self.yolo_weight = str(Path(yolo_weight).resolve())

        print("=" * 60)
        print("Memuat model YOLO")
        print("Path :", self.yolo_weight)
        print("=" * 60)

        if not Path(self.yolo_weight).exists():
            raise FileNotFoundError(
                f"File model YOLO tidak ditemukan:\n{self.yolo_weight}"
            )

        try:
            self.yolo_model = YOLO(self.yolo_weight)
            print("Model YOLO berhasil dimuat")

        except Exception:
            print("Gagal memuat model YOLO")
            print(traceback.format_exc())
            raise
    def segment_braille(self, image_path):
        """Segmentasi sel Braille dari gambar."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")

        results = self.yolo_model.predict(image, conf=self.conf, max_det=9999)
        boxes = results[0].boxes
        list_boxes = parse_xywh_and_class(boxes)
        return results, list_boxes

    def get_distance(self, xs):
        """Hitung jarak horizontal antar kotak dalam satu baris."""
        if len(xs) < 2:
            return np.array([]), 0

        distances = np.diff(xs)
        common_distance = Counter(distances).most_common(1)[0][0]
        return distances, common_distance

    def get_box_properties(self, row):
        xs = [box[0] for box in row.astype(int)]
        classes = [box[-1] for box in row.astype(int)]
        distances, common = self.get_distance(xs)
        return xs, classes, distances, common

    def clean_bboxes(self, boxes):
        """Hilangkan bounding box yang tumpang tindih."""
        cleaned_boxes = []

        for row in boxes:
            if len(row) <= 1:
                cleaned_boxes.append(row)
                continue

            xs, _, distances, common = self.get_box_properties(row)
            if common == 0:
                cleaned_boxes.append(row)
                continue

            delete_indices = []
            for j in range(1, len(xs)):
                if distances[j - 1] < common / 2:
                    delete_indices.append(j - 1)

            if delete_indices:
                row = np.delete(row, delete_indices, axis=0)
            cleaned_boxes.append(row)

        return cleaned_boxes
