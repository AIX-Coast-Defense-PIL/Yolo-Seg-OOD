import os
from PyQt5.QtWidgets import *


class ResultWindow(QMainWindow):
    def __init__(self, task):
        super().__init__()
        self.setWindowTitle('Result Window')

        layout = QVBoxLayout()
        result_label = QLabel('실행 결과', self)
        layout.addWidget(result_label)

        # 쉘 파일 실행 후 결과 텍스트 가져오기
        # 여기서는 단순히 쉘 파일 이름을 텍스트로 표시하는 예시입니다.
        result_label.setText(task)

        # 창 닫기 버튼 생성
        close_button = QPushButton('닫기', self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
