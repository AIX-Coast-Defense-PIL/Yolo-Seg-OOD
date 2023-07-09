import os
import sys
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

root_path = os.path.join(os.getcwd(), os.pardir) if '/yolov7' in os.getcwd() else os.getcwd()
root_path = root_path if 'Yolo-Seg-OOD' in root_path else os.path.join(root_path, 'Yolo-Seg-OOD')
sys.path.append(root_path)


class LabelingTool(QWidget):
    def __init__(self):
        super().__init__()
        self.image_dir = './ood/datasets/boundary_data/images'
        json_path = './ood/datasets/boundary_data/yolov7_preds/yolov7_preds_refined.json'
        if os.path.exists(json_path):
            with open(json_path, 'rb') as file:
                self.json_dict = json.load(file)
            if not len(self.json_dict):
                self.json_dict = None
        else: self.json_dict = None
        self.current_image_index = 0
        self.img_size = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle("이미지 라벨링 도구")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.img_size, self.img_size)

        self.reference_label = QLabel(self)
        self.reference_label.setFixedSize(self.img_size, self.img_size)

        self.no_image = QLabel('test')
        self.no_image.setFixedSize(self.img_size, self.img_size)
        self.no_image.setAlignment(Qt.AlignCenter)

        self.prev_button = QPushButton("◀", self)
        self.next_button = QPushButton("▶", self)
        self.known_button = QPushButton("Known", self)
        self.unknown_button = QPushButton("Unknown", self)
        self.save_button = QPushButton("저장", self)

        self.prev_button.setFixedWidth(40)
        self.next_button.setFixedWidth(40)
        self.known_button.setFixedWidth(90)
        self.unknown_button.setFixedWidth(90)
        self.save_button.setFixedWidth(90)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.known_button.clicked.connect(lambda: self.label_image("Known"))
        self.unknown_button.clicked.connect(lambda: self.label_image("Unknown"))

        image_layout = QHBoxLayout()
        if self.json_dict is not None:
            image_layout.addWidget(self.image_label)
            image_layout.addWidget(self.reference_label)
            self.show_current_image()
        else:
            image_layout.addWidget(self.no_image)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.known_button.setEnabled(False)
            self.unknown_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        label_layout = QVBoxLayout()
        label_layout.addStretch(5)
        label_layout.addWidget(self.known_button)
        label_layout.addWidget(self.unknown_button)
        label_layout.addStretch(1)
        label_layout.addLayout(button_layout)
        label_layout.addStretch(4)
        label_layout.addWidget(self.save_button)

        spacer_item = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Preferred)
        label_layout.addSpacerItem(spacer_item)


        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(label_layout)

        self.setLayout(main_layout)

    def show_current_image(self):
        if self.json_dict is not None:
            img_id = self.json_dict[self.current_image_index]["image_id"]
            image_path = os.path.join(self.image_dir, img_id + '.jpg')
            bbox_xyxy = self.json_dict[self.current_image_index]["bbox"]
            bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]

            pixmap = QPixmap(image_path)
            cropped_pixmap = pixmap.copy(*bbox_xywh)

            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(*bbox_xywh)
            painter.end()

            self.image_label.setPixmap(pixmap.scaled(self.img_size, self.img_size))
            self.reference_label.setPixmap(cropped_pixmap.scaled(self.img_size, self.img_size, Qt.KeepAspectRatio))
            
        if self.current_image_index == 0:
            self.prev_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(True)

        if self.current_image_index == len(self.json_dict) - 1:
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
    
    def update_button_state(self):
        if self.json_dict is not None:
            is_known = self.json_dict[self.current_image_index]["is_known"]
            if is_known == 1:
                self.known_button.setDown(True)
                self.unknown_button.setDown(False)
            elif is_known == 0:
                self.known_button.setDown(False)
                self.unknown_button.setDown(True)
            else:
                self.known_button.setDown(False)
                self.unknown_button.setDown(False)
        else:
            self.known_button.setDown(False)
            self.unknown_button.setDown(False)

    def show_previous_image(self):
        if self.json_dict is not None:
            self.current_image_index -= 1
            if self.current_image_index < 0:
                self.current_image_index = len(self.json_dict) - 1
            self.update_button_state()
            self.show_current_image()

    def show_next_image(self):
        if self.json_dict is not None:
            self.current_image_index += 1
            if self.current_image_index >= len(self.json_dict):
                self.current_image_index = 0
            self.update_button_state()
            self.show_current_image()

    def label_image(self, label):
        if self.json_dict is not None:
            image_id = self.json_dict[self.current_image_index]["image_id"]

            if label == "Known":
                is_known = 1
                self.known_button.setDown(True)
                self.unknown_button.setDown(False)
            elif label == "Unknown":
                is_known = 0
                self.known_button.setDown(False)
                self.unknown_button.setDown(True)

            for item in self.json_dict:
                if item["image_id"] == image_id:
                    item["is_known"] = is_known
                    break

            # json 파일 저장
            print(self.json_dict)
            # json_path = './ood/datasets/boundary_data/yolov7_preds/yolov7_preds_refined.json'
            # with open(json_path, 'w') as file:
            #     json.dump(self.json_dict, file)

            # print(f"Image labeled as {label}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    labeling_tool = LabelingTool()
    labeling_tool.show()
    sys.exit(app.exec_())
