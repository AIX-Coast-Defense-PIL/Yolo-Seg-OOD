import os
import sys
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from utils import WINDOW_TITLE

class ShellScriptThread(QThread):
    progress_updated = pyqtSignal(int)
    output_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    script_finished = pyqtSignal()

    def __init__(self, task, command):
        super().__init__()
        self.task = task
        self.command = command

    def run(self):
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        self.completed_lines = 0
        self.start_train = False

        while True:
            line = process.stdout.readline()
            if not line:
                break

            ## Update Terminal Output
            line = line.decode().strip()
            self.output_updated.emit(line)
            
            ## Update Progress
            if self.task == 'seg':
                self.update_grogress_seg(line)

            ## Error Message
            if "error" in line.lower():
                self.error_occurred.emit(line)

        process.wait()
        self.script_finished.emit()
    
    def update_grogress_seg(self, line):
        if 'The number of epochs:' in line:
            self.num_epochs = line.split('  ')[1]
        
        if line.split(' ')[0] == 'Training:':
            self.start_train = True
        
        if self.start_train:
            if 'Epoch ' in line:
                t = line.split('Epoch ')[1][:10]
                epoch, perc = t.split(':')
                perc = perc.split('%')[0].lstrip()
                if int(perc) != 0:
                    progress = 5 + int((int(epoch) + (int(perc)-1)/100) / int(self.num_epochs) * 95)
                    self.progress_updated.emit(progress)
            elif 'Train Finish!' in line:
                self.progress_updated.emit(100)
        else:
            self.completed_lines += 1
            progress = 5 * (1 - (1.6)**(-self.completed_lines/5))
            self.progress_updated.emit(progress)



class ResultWindow(QWidget):
    def __init__(self, task, script_path):
        super().__init__()
        self.task = task
        self.script_path = script_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOW_TITLE[self.task] + ' Results')

        layout = QVBoxLayout(self)

        progress_layout = QVBoxLayout()
        progress_layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("진행 상황:")
        progress_layout.addWidget(self.label)

        bar_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        bar_layout.addWidget(self.progress_bar)

        self.toggle_button = QPushButton("▼ 자세히")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        bar_layout.addWidget(self.toggle_button)

        progress_layout.addLayout(bar_layout)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.hide()
        progress_layout.addWidget(self.output_text)

        layout.addLayout(progress_layout)

        self.close_button = QPushButton("창 닫기")
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button)

        self.thread = None

        self.toggle_button.clicked.connect(self.toggle_output)
        self.close_button.clicked.connect(self.close)

        self.thread_error_occurred = False

        self.resize(600, 200)
        self.center(40)


    def center(self, mv=0):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp + QPoint(mv,mv))
        self.move(qr.topLeft())

    def start_script(self):
        self.thread = ShellScriptThread(self.task, f". {self.script_path}")
        self.thread.progress_updated.connect(self.update_progress) 
        self.thread.output_updated.connect(self.append_output)
        self.thread.error_occurred.connect(self.show_error_message)
        self.thread.script_finished.connect(self.script_finished)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_output(self, text):
        self.output_text.append(text)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    def update_output(self):
        self.output_text.ensureCursorVisible()

    def toggle_output(self):
        if self.toggle_button.isChecked():
            self.output_text.show()
            self.toggle_button.setText("▲ 출력 접기")
            self.resize(600, 400)
        else:
            self.output_text.hide()
            self.toggle_button.setText("▼ 자세히")
            self.resize(600, 100)

    def show_error_message(self, error):
        self.thread_error_occurred = True
        QMessageBox.critical(self, "오류 발생", error)

    def script_finished(self):
        self.close_button.setEnabled(True)
        
        if '_edit.sh' in self.script_path:
            os.remove(self.script_path)

            
if __name__ == '__main__':
    app = QApplication(sys.argv)

    result_window = ResultWindow('seg', './shell/train_seg_temp.sh')
    result_window.show()
    result_window.start_script()

    sys.exit(app.exec_())