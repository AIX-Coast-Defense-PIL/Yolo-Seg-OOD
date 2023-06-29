import os
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from utils import WINDOW_TITLE

class ShellScriptThread(QThread):
    progress_updated = pyqtSignal(int)  # 진행률 업데이트 시그널
    output_updated = pyqtSignal(str)  # 터미널 출력 업데이트 시그널
    error_occurred = pyqtSignal(str)  # 오류 발생 시그널
    script_finished = pyqtSignal()  # 스크립트 실행 완료 시그널

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        total_lines = 0  # 전체 출력 라인 수 추적
        completed_lines = 0  # 완료된 출력 라인 수 추적

        while True:
            line = process.stdout.readline()
            if not line:
                break

            line = line.decode().strip()
            self.output_updated.emit(line)  # 터미널 출력 업데이트

            # 여기에 다른 처리를 추가하세요.
            completed_lines += 1

            if "error" in line.lower():  # 오류 메시지가 포함된 행이 있는지 확인
                self.error_occurred.emit(line)  # 오류 발생 시그널 발생

        process.wait()
        self.script_finished.emit()  # 스크립트 실행 완료 시그널 발생


class ResultWindow(QWidget):
    def __init__(self, task, script_path):
        super().__init__()
        self.task = task
        self.script_path = script_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOW_TITLE[self.task] + ' Results')

        layout = QVBoxLayout(self)
        self.label = QLabel("진행 상황:")
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)  # 읽기 전용으로 설정
        layout.addWidget(self.output_text)

        self.toggle_button = QPushButton("출력 접기")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        layout.addWidget(self.toggle_button)

        self.close_button = QPushButton("창 닫기")
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button)

        self.thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)  # 출력 업데이트 타이머에 연결

        self.toggle_button.clicked.connect(self.toggle_output)  # 출력 토글 버튼에 클릭 이벤트 연결
        self.close_button.clicked.connect(self.close)  # 창 닫기 버튼에 클릭 이벤트 연결

        self.thread_error_occurred = False

        self.resize(600, 400)
        self.center(40)
        # self.show()

    def center(self, mv=0):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp + QPoint(mv,mv))
        self.move(qr.topLeft())

    def start_script(self):
        self.thread = ShellScriptThread(f". {self.script_path}")  # 실행할 셸 스크립트 파일 이름으로 변경해야 함
        self.thread.progress_updated.connect(self.update_progress)  # 진행률 업데이트 시그널 연결
        self.thread.output_updated.connect(self.append_output)  # 터미널 출력 업데이트 시그널 연결
        self.thread.error_occurred.connect(self.show_error_message)  # 오류 발생 시그널 연결
        self.thread.script_finished.connect(self.script_finished)  # 스크립트 실행 완료 시그널 연결
        self.thread.start()
        self.timer.start(100)  # 0.1초마다 출력 업데이트 타이머 시작

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_output(self, text):
        self.output_text.append(text)  # 텍스트 박스에 텍스트 추가

    def update_output(self):
        self.output_text.ensureCursorVisible()  # 텍스트 박스 스크롤 자동 조정

    def toggle_output(self):
        if self.toggle_button.isChecked():
            self.output_text.hide()
            self.toggle_button.setText("출력 펼치기")
        else:
            self.output_text.show()
            self.toggle_button.setText("출력 접기")

    def show_error_message(self, error):
        self.thread_error_occurred = True
        QMessageBox.critical(self, "오류 발생", error)

    def script_finished(self):
        self.timer.stop()
        self.toggle_button.setEnabled(False)
        self.close_button.setEnabled(True)
        
        if '_edit.sh' in self.script_path:
            os.remove(self.script_path)