import json
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from .segmentation import BrailleSegmentation


class BrailleClassifier:
    def __init__(
        self,
        model_path="weights/cnn_v1.hdf5",
        json_path="utils/class_labels.json",
        symbols_path="utils/braille_symbols.json",
        numbers_path="utils/braille_numbers.json",
        yolo_weight="weights/yolov8_braille.pt",
    ):
        self.segmenter = BrailleSegmentation(yolo_weight=yolo_weight)
        self.dim = (70, 100)
        self.json_path = json_path
        self.symbols_path = symbols_path
        self.numbers_path = numbers_path
        self.model = load_model(model_path)
        self.vowels = {"a", "i", "u", "e", "o"}

    def import_class_file(self):
        with open(self.json_path, encoding="utf-8") as json_file:
            return json.load(json_file)

    def get_class(self, prediction, class_labels):
        for key, value in class_labels.items():
            if prediction == value:
                return key
        return ""

    def convert_symbols(self, symbols):
        with open(self.symbols_path, encoding="utf-8") as path:
            braille_symbols = json.load(path)
        return braille_symbols.get(symbols, symbols)

    def convert_numbers(self, text):
        with open(self.numbers_path, encoding="utf-8") as data:
            numbers_dict = json.load(data)
        return numbers_dict.get(text, text)

    def get_times_new_roman_bold(self, size=16):
        """Mengambil font Times New Roman Bold setara 12pt jika tersedia."""
        font_candidates = [
            Path("C:/Windows/Fonts/timesbd.ttf"),
            Path("C:/Windows/Fonts/times.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ]

        for font_path in font_candidates:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size=size)

        return ImageFont.load_default()

    def draw_centered_text(self, image, text, start_point, end_point):
        """Menulis label pada grid gambar deteksi dengan Times New Roman 12pt bold."""
        if not text:
            return

        x1, y1 = start_point
        x2, y2 = end_point
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = self.get_times_new_roman_bold(size=16)
        label = str(text).upper()

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1 + max(0, (x2 - x1 - text_width) // 2)
        text_y = y1 + int((y2 - y1) * 0.62) - text_height // 2

        draw.text((text_x + 1, text_y + 1), label, fill=(255, 255, 255), font=font)
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

        image[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def draw_raised_box(self, image, start_point, end_point):
        """Membuat kotak grid pada gambar deteksi terlihat lebih timbul."""
        x1, y1 = start_point
        x2, y2 = end_point

        if x2 <= x1 or y2 <= y1:
            return

        overlay = image.copy()
        shadow_offset = 4

        sx1 = min(image.shape[1] - 1, x1 + shadow_offset)
        sy1 = min(image.shape[0] - 1, y1 + shadow_offset)
        sx2 = min(image.shape[1], x2 + shadow_offset)
        sy2 = min(image.shape[0], y2 + shadow_offset)
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.16, image, 0.84, 0, image)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.line(image, (x1, y1), (x2, y1), (255, 255, 255), 1)
        cv2.line(image, (x1, y1), (x1, y2), (255, 255, 255), 1)

    def preprocess_cells(self, cell):
        if cell is None or cell.size == 0:
            return None

        braille_letter = PIL.Image.fromarray(cell)
        processed_img = braille_letter.resize(self.dim)
        processed_img = img_to_array(processed_img)
        processed_img = processed_img / 255.0
        processed_img = processed_img.reshape(1, 100, 70, 3)
        return processed_img

    def convert_coordinate(self, box, image_shape=None):
        xcent, ycent, w, h = box[:4].astype(int)
        top = ycent - int(h / 2) - 5
        left = xcent - int(w / 2) - 5
        bottom = ycent + int(h / 2) + 5
        right = xcent + int(w / 2) + 5

        if image_shape is not None:
            img_h, img_w = image_shape[:2]
            left = max(0, min(left, img_w - 1))
            right = max(0, min(right, img_w))
            top = max(0, min(top, img_h - 1))
            bottom = max(0, min(bottom, img_h))

        start_point = (left, top)
        end_point = (right, bottom)
        return start_point, end_point, xcent, ycent

    def classify_braille(self, model, image, class_labels):
        label = model.predict(image, verbose=0)
        label = label.argmax(axis=-1)[0]
        label = self.get_class(label, class_labels)
        label = self.convert_symbols(label)
        return label

    def get_raw_texts(self, boxes, image, model, class_labels):
        texts = []

        for row in boxes:
            row_text = []
            for box in row:
                start_point, end_point, _, _ = self.convert_coordinate(box, image.shape)
                x1, y1 = start_point
                x2, y2 = end_point
                cropped_image = image[y1:y2, x1:x2]
                cropped_image = self.preprocess_cells(cropped_image)

                if cropped_image is None:
                    row_text.append("")
                    continue

                label = self.classify_braille(model, cropped_image, class_labels)
                row_text.append(label)

            texts.append(row_text)

        return texts

    def get_spaces(self, boxes, raw_texts):
        for i, row in enumerate(boxes):
            if len(row) < 2:
                continue

            space_index = []
            _, _, distances, common = self.segmenter.get_box_properties(row)
            if common == 0:
                continue

            for j in range(len(raw_texts[i])):
                if j > 0 and distances[j - 1] > common * 1.5:
                    space_index.append(j)

            space_index.reverse()
            for j in space_index:
                raw_texts[i].insert(j, " ")

        return raw_texts

    def join_texts(self, spaced_texts):
        joined_texts = []

        for row in spaced_texts:
            row_text = "".join(row).replace("^", " ")
            words = row_text.split(" ")
            converted_words = []

            for word in words:
                if word == "":
                    converted_words.append("")
                    continue

                if word[0] == "#":
                    chars = list(word)
                    chars[0] = " "
                    converted_words.append("".join(self.convert_numbers(char) for char in chars))
                else:
                    converted_words.append(word)

            joined_texts.append(converted_words)

        return joined_texts

    def is_vowel(self, text):
        return str(text).lower() in self.vowels

    def is_simple_letter(self, text):
        return isinstance(text, str) and len(text) == 1 and text.isalpha()

    def merge_boxes_xywh(self, selected_boxes):
        if len(selected_boxes) == 1:
            return selected_boxes[0]

        corners = []
        confs = []
        classes = []
        for box in selected_boxes:
            xcent, ycent, w, h = box[:4]
            x1 = xcent - (w / 2)
            y1 = ycent - (h / 2)
            x2 = xcent + (w / 2)
            y2 = ycent + (h / 2)
            corners.append((x1, y1, x2, y2))
            confs.append(box[4] if len(box) > 4 else 0)
            classes.append(box[5] if len(box) > 5 else 0)

        x1 = min(v[0] for v in corners)
        y1 = min(v[1] for v in corners)
        x2 = max(v[2] for v in corners)
        y2 = max(v[3] for v in corners)

        xcent = (x1 + x2) / 2
        ycent = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        conf = float(np.mean(confs)) if confs else 0.0
        cls = classes[0] if classes else 0.0
        return np.array([xcent, ycent, w, h, conf, cls], dtype=float)

    def split_row_to_words(self, row_boxes, row_texts):
        if len(row_boxes) == 0:
            return []

        _, _, distances, common = self.segmenter.get_box_properties(row_boxes)
        threshold = common * 1.5 if common else None
        words = []
        current_word = []

        for idx, box in enumerate(row_boxes):
            label = row_texts[idx] if idx < len(row_texts) else ""
            if idx > 0 and threshold is not None and distances[idx - 1] > threshold:
                if current_word:
                    words.append(current_word)
                current_word = []

            if label and label not in (" ", "^"):
                current_word.append((box, label))

        if current_word:
            words.append(current_word)

        return words

    def syllabify_word_items(self, word_items):
        if not word_items:
            return []

        grouped_items = []
        i = 0
        while i < len(word_items):
            current_box, current_label = word_items[i]
            current_label_str = str(current_label).lower()

            # Aturan sederhana untuk kelas 1 SD:
            # gabungkan dua karakter jika polanya konsonan + vokal => ba, bu, be, dst.
            if (
                self.is_simple_letter(current_label_str)
                and not self.is_vowel(current_label_str)
                and i + 1 < len(word_items)
            ):
                next_box, next_label = word_items[i + 1]
                next_label_str = str(next_label).lower()
                if self.is_simple_letter(next_label_str) and self.is_vowel(next_label_str):
                    merged_box = self.merge_boxes_xywh([current_box, next_box])
                    grouped_items.append((merged_box, current_label_str + next_label_str))
                    i += 2
                    continue

            grouped_items.append((current_box, current_label_str))
            i += 1

        return grouped_items

    def build_syllable_groups(self, boxes, raw_texts):
        grouped_boxes = []
        grouped_labels = []
        grouped_words = []
        grouped_word_units = []

        for row_index, row_boxes in enumerate(boxes):
            row_texts = raw_texts[row_index] if row_index < len(raw_texts) else []
            word_items_list = self.split_row_to_words(row_boxes, row_texts)

            row_group_boxes = []
            row_group_labels = []
            row_word_texts = []
            row_word_units = []

            for word_items in word_items_list:
                syllable_items = self.syllabify_word_items(word_items)
                syllable_texts = []
                for merged_box, syllable_text in syllable_items:
                    row_group_boxes.append(merged_box)
                    row_group_labels.append(syllable_text)
                    syllable_texts.append(syllable_text)

                if syllable_texts:
                    row_word_texts.append("".join(syllable_texts))
                    row_word_units.append(syllable_texts)

            grouped_boxes.append(np.array(row_group_boxes, dtype=float) if row_group_boxes else np.empty((0, 6), dtype=float))
            grouped_labels.append(row_group_labels)
            grouped_words.append(row_word_texts)
            grouped_word_units.append(row_word_units)

        return grouped_boxes, grouped_labels, grouped_words, grouped_word_units

    def draw_final(self, image, boxes, raw_texts):
        """Menggambar grid deteksi pada gambar kanan dengan efek timbul."""
        for row_index, row in enumerate(boxes):
            for col_index, box in enumerate(row):
                start_point, end_point, _, _ = self.convert_coordinate(box, image.shape)
                text = ""
                if row_index < len(raw_texts) and col_index < len(raw_texts[row_index]):
                    text = raw_texts[row_index][col_index]

                self.draw_raised_box(image, start_point, end_point)

        return image

    def build_display_text(self, joined_texts):
        """Membuat teks hasil prediksi final berupa kata."""
        lines = []
        for row in joined_texts:
            line = " ".join(word for word in row if word != "").strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def build_speech_text(self, word_units_by_row):
        """
        Membuat naskah suara pembelajaran yang singkat.
        Contoh: pa + k = pak. bu + ku = buku. i + bu = ibu.
        """
        spoken_parts = []

        for row_units in word_units_by_row:
            for syllables in row_units:
                clean_syllables = []
                for item in syllables:
                    syllable = str(item).lower().strip()
                    if not syllable or syllable in (" ", "^", "#"):
                        continue
                    # Hindari tanda hasil salah prediksi agar naskah suara tidak membingungkan.
                    syllable = "".join(char for char in syllable if char.isalpha() or char.isdigit())
                    if syllable:
                        clean_syllables.append(syllable)

                if not clean_syllables:
                    continue

                final_word = "".join(clean_syllables)
                if len(clean_syllables) > 1:
                    joined_syllables = " + ".join(clean_syllables)
                    spoken_parts.append(f"{joined_syllables} = {final_word}")
                else:
                    spoken_parts.append(final_word)

        return "\n".join(spoken_parts)

    def speech_label(self, label):
        if label in ("", " ", "^"):
            return ""
        if label == "#":
            return "tanda angka"
        return str(label).lower()

    def build_detected_cells(self, boxes, raw_texts, image_shape):
        """Data posisi grid untuk animasi melayang pada gambar kanan di halaman web."""
        img_h, img_w = image_shape[:2]
        detected_cells = []

        if img_h <= 0 or img_w <= 0:
            return detected_cells

        for row_index, row in enumerate(boxes):
            for col_index, box in enumerate(row):
                start_point, end_point, _, _ = self.convert_coordinate(box, image_shape)
                x1, y1 = start_point
                x2, y2 = end_point

                label = ""
                if row_index < len(raw_texts) and col_index < len(raw_texts[row_index]):
                    label = raw_texts[row_index][col_index]

                if not label or label in (" ", "^"):
                    continue

                detected_cells.append(
                    {
                        "left": round((x1 / img_w) * 100, 4),
                        "top": round((y1 / img_h) * 100, 4),
                        "width": round(((x2 - x1) / img_w) * 100, 4),
                        "height": round(((y2 - y1) / img_h) * 100, 4),
                        "text": str(label).lower(),
                        "speak": self.speech_label(label),
                    }
                )

        return detected_cells

    def recognize_braille(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")

        class_labels = self.import_class_file()
        _, list_boxes = self.segmenter.segment_braille(image_path)

        if not list_boxes:
            raise RuntimeError("Tidak ada titik/karakter Braille yang terdeteksi pada gambar.")

        boxes = self.segmenter.clean_bboxes(list_boxes)
        raw_texts = self.get_raw_texts(boxes, image, self.model, class_labels)

        # Tahap baru:
        # 1) YOLO tetap dipakai untuk segmentasi/grid per karakter.
        # 2) Hasil karakter kemudian digabung menjadi suku kata sederhana (CV), misalnya ba, bu, be.
        # 3) Suku kata lalu digabung kembali menjadi kata, misalnya i + bu -> ibu.
        grouped_boxes, grouped_labels, grouped_words, grouped_word_units = self.build_syllable_groups(boxes, raw_texts)

        # Output karakter tetap dibuat, supaya kemampuan lama A-Z tidak hilang.
        raw_texts_for_grid = [row[:] for row in raw_texts]
        spaced_character_texts = self.get_spaces(boxes, [row[:] for row in raw_texts])
        joined_character_texts = self.join_texts(spaced_character_texts)
        character_text = self.build_display_text(joined_character_texts)
        character_cells = self.build_detected_cells(boxes, raw_texts_for_grid, image.shape)

        # Output tambahan untuk suku kata/kata.
        syllable_text = self.build_display_text(grouped_words)
        syllable_cells = self.build_detected_cells(grouped_boxes, grouped_labels, image.shape)
        speech_text = self.build_speech_text(grouped_word_units)

        # Gambar disimpan tanpa kotak permanen. Kotak karakter/suku kata digambar oleh overlay HTML
        # supaya pengguna dapat memilih mode grid yang ingin dilihat.
        final_image = image.copy()
        return final_image, character_text, syllable_text, speech_text, character_cells, syllable_cells
