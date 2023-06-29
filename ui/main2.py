import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget, QVBoxLayout, QWidget, QLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window')

        tab_widget = QTabWidget(self)
        self.setCentralWidget(tab_widget)

        # 탭 생성 및 추가
        tab1 = TabWidget('shell_script1.sh')
        tab_widget.addTab(tab1, '탭 1')

        tab2 = TabWidget('shell_script2.sh')
        tab_widget.addTab(tab2, '탭 2')

        tab3 = TabWidget('shell_script3.sh')
        tab_widget.addTab(tab3, '탭 3')


# 탭 내용을 담은 위젯
class TabWidget(QWidget):
    def __init__(self, shell_file):
        super().__init__()
        self.shell_file = shell_file

        layout = QVBoxLayout()
        self.result_label = QLabel('실행 결과', self)
        layout.addWidget(self.result_label)

        start_button = QPushButton('시작', self)
        start_button.clicked.connect(self.open_result_window)
        layout.addWidget(start_button)

        self.setLayout(layout)

    # 결과 창 열기
    def open_result_window(self):
        self.result_window = ResultWindow(self.shell_file)
        self.result_window.show()


# 결과 창 클래스
class ResultWindow(QMainWindow):
    def __init__(self, shell_file):
        super().__init__()
        self.setWindowTitle('Result Window')

        layout = QVBoxLayout()
        result_label = QLabel('실행 결과', self)
        layout.addWidget(result_label)

        # 쉘 파일 실행 후 결과 텍스트 가져오기
        # 여기서는 단순히 쉘 파일 이름을 텍스트로 표시하는 예시입니다.
        result_label.setText(shell_file)

        # 창 닫기 버튼 생성
        close_button = QPushButton('닫기', self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
