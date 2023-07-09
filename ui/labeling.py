import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class LabelingTool(QWidget):
    def __init__(self):
        super().__init__()
        self.image_directory = './ood/datasets/boundary_data/images'
        self.image_list = []
        self.current_image_index = 0
        self.img_size = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle("이미지 라벨링 도구")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.img_size, self.img_size)

        self.reference_label = QLabel(self)
        self.reference_label.setFixedSize(self.img_size, self.img_size)

        self.prev_button = QPushButton("◀", self)
        self.next_button = QPushButton("▶", self)
        self.known_button = QPushButton("Known", self)
        self.unknown_button = QPushButton("Unknown", self)

        self.prev_button.setFixedWidth(40)
        self.next_button.setFixedWidth(40)
        self.known_button.setFixedWidth(90)
        self.unknown_button.setFixedWidth(90)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.known_button.clicked.connect(lambda: self.label_image("Known"))
        self.unknown_button.clicked.connect(lambda: self.label_image("Unknown"))

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.reference_label)

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

        spacer_item = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Preferred)
        label_layout.addSpacerItem(spacer_item)


        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(label_layout)

        self.setLayout(main_layout)
        self.load_images()
        self.show_current_image()

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 열기")
        if directory:
            self.image_directory = directory
            self.load_images()
            self.show_current_image()

    def load_images(self):
        self.image_list = []
        if self.image_directory:
            image_formats = (".png", ".jpg", ".jpeg", ".bmp")
            for file_name in os.listdir(self.image_directory):
                if file_name.lower().endswith(image_formats):
                    self.image_list.append(os.path.join(self.image_directory, file_name))

    def show_current_image(self):
        if self.image_list:
            image_path = self.image_list[self.current_image_index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.img_size, self.img_size))

            # 참조 이미지를 두 번째 QLabel에 표시합니다.
            self.reference_label.setPixmap(pixmap.scaled(self.img_size, self.img_size))

    def show_previous_image(self):
        if self.image_list:
            self.current_image_index -= 1
            if self.current_image_index < 0:
                self.current_image_index = len(self.image_list) - 1
            self.show_current_image()

    def show_next_image(self):
        if self.image_list:
            self.current_image_index += 1
            if self.current_image_index >= len(self.image_list):
                self.current_image_index = 0
            self.show_current_image()

    def label_image(self, label):
        if self.image_list:
            image_path = self.image_list[self.current_image_index]
            image_file_name = os.path.basename(image_path)

            if label == "Known":
                file_name = "known_images.txt"
            elif label == "Unknown":
                file_name = "unknown_images.txt"

            with open(file_name, "a") as file:
                file.write(image_file_name + "\n")

            print(f"Image labeled as {label}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    labeling_tool = LabelingTool()
    labeling_tool.show()
    sys.exit(app.exec_())
