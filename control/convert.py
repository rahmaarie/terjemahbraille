import json
import numpy as np
import torch


def convert_to_braille_unicode(str_input: str, path: str = "utils/braille_map.json") -> str:
    with open(path, "r", encoding="utf-8") as fl:
        data = json.load(fl)
    return data.get(str_input, str_input)


def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """Ubah hasil deteksi YOLO menjadi list baris Braille yang sudah diurutkan."""
    jumlah_box = len(boxes)
    if jumlah_box == 0:
        return []

    new_boxes = np.zeros((jumlah_box, 6), dtype=float)
    new_boxes[:, :4] = boxes.xywh.cpu().numpy()
    new_boxes[:, 4] = boxes.conf.cpu().numpy()
    new_boxes[:, 5] = boxes.cls.cpu().numpy()

    # Urutkan dari atas ke bawah.
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    if jumlah_box == 1:
        return [new_boxes]

    # Pecah menjadi beberapa baris berdasarkan jarak koordinat y.
    tinggi_rata_rata = np.mean(new_boxes[:, 3])
    y_threshold = max(tinggi_rata_rata / 2, 1)
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]

    rows = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for row in rows:
        row = row[row[:, 0].argsort()]
        boxes_return.append(row)

    return boxes_return
