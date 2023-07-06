import os
import sys
import time
import subprocess
import numpy as np
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
        self.val_min, self.val_max = 0, 5
        self.num_datasets = 0

        while True:
            line = process.stdout.readline()
            if not line:
                break

            ## Update Terminal Output
            line = line.decode().strip()
            self.output_updated.emit(line)
            
            ## Update Progress
            if self.task == 'seg':
                self.update_grogress_train_seg(line)
            elif self.task == 'ood':
                if 'infer_yolo' in self.command:
                    self.update_grogress_infer_yolo(line)
                elif 'filter_yolo' in self.command:
                    self.update_grogress_filter_preds(line)
                elif 'train_ood_cluster' in self.command:
                    self.update_grogress_train_cluster(line)

            ## Error Message
            if "error" in line.lower():
                self.error_occurred.emit(line)
            
        process.wait()
        self.script_finished.emit()
    
    def update_grogress_train_seg(self, line):
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
    
    def update_grogress_infer_yolo(self, line):
        ## infer_yolo.sh
        if 'if param.grad is not None:' in line:
            self.val_min, self.val_max = 5, 30
            self.completed_lines = 0
            self.progress_updated.emit(5)
        
        elif 'corrupted: ' in line:
            self.val_min, self.val_max = 30, 100
            self.completed_lines = 0
            self.progress_updated.emit(30)

        elif 'mAP@.5:.95:' in line:
            t = line.split('mAP@.5:.95:')[1][:10]
            perc = int(t.split('%')[0].lstrip())
            self.progress_updated.emit(30 + int(perc * 0.7))
        
        elif 'YOLO-v7 prediction Done!' in line:
            self.val_min = 100
            self.start_infer_yolo = False
            self.progress_updated.emit(100)
        
        else:
            if self.val_min != 100:
                progress = int(self.val_min + (self.val_max - self.val_min) / (1 + np.exp(-0.7 * self.completed_lines + 5)))
                self.progress_updated.emit(progress)

        self.completed_lines += 1
    
    def update_grogress_filter_preds(self, line):
        ## filter_yolo_preds.sh
        if 'The number of test datasets:' in line:
            self.num_datasets = int(line.split('  ')[1])

        elif (self.num_datasets != 0) and (f'/{self.num_datasets}' in line):
            t = int(line.split(f'/{self.num_datasets}')[0])
            if t != 0:
                progress = int((t / self.num_datasets) * 100)
                if progress != 100: self.progress_updated.emit(progress)
            
        elif 'YOLO-v7 prediction filtering Done!' in line:
            self.start_filter_preds = False
            self.progress_updated.emit(100)

    def update_grogress_train_cluster(self, line):
        ## train_ood_cluster.sh
        if 'Loaded thresholds from' in line:
            self.progress_updated.emit(50)
        elif 'ood thresholds : ' in line:
            self.progress_updated.emit(80)
        
        elif 'OOD cluster (K-Means) train Done!' in line:
            self.start_train_cluster = False
            self.progress_updated.emit(100)


class ResultWindow(QWidget):
    def __init__(self, task, num_script):
        super().__init__()
        self.ood_progress = 0
        self.task = task
        self.num_script = num_script
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOW_TITLE[self.task] + ' Results')

        layout = QVBoxLayout(self)

        progress_layout = QVBoxLayout()
        progress_layout.setAlignment(Qt.AlignCenter)

        if self.task == 'ood':
            self.label = QLabel("<span style='color: black'>진행 상황 </span>"
                            f"<span style='color: blue'>(1/{self.num_script}):</span>")
        else:
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

        self.complete_text = QLabel('')
        self.complete_text.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.complete_text)

        layout.addLayout(progress_layout)

        self.close_button = QPushButton("창 닫기")
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button)

        self.thread = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)

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

    def start_script(self, script_path):
        self.script_index = 0
        self.script_path = script_path

        self.execute_script(self.script_path[self.script_index])

    def execute_script(self, script):
        self.script_thread = ShellScriptThread(self.task, f". {script}")
        self.script_thread.progress_updated.connect(self.update_progress)
        self.script_thread.output_updated.connect(self.append_output)
        self.script_thread.error_occurred.connect(self.show_error_message)
        self.script_thread.script_finished.connect(self.script_finished)
        self.script_thread.start()
        self.timer.start(100)
        
        QApplication.processEvents()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

        if value == 100:
            time.sleep(3)
            self.ood_progress += 1
            if self.ood_progress != self.num_script:
                self.label.setText("<span style='color: black'>진행 상황 </span>"
                                f"<span style='color: blue'>({self.ood_progress+1}/{self.num_script}):</span>")
                self.progress_bar.setValue(0)
            else:
                self.complete_text.setText("학습이 완료되었습니다.")

    def append_output(self, text):
        self.output_text.append(text)
        self.output_text.ensureCursorVisible()

    def update_output(self):
        self.output_text.ensureCursorVisible()
        QApplication.processEvents()

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
        self.timer.stop()

        if self.script_index == len(self.script_path) - 1:
            self.close_button.setEnabled(True)
        else:
            self.script_index += 1
            self.execute_script(self.script_path[self.script_index])
    
            
if __name__ == '__main__':
    app = QApplication(sys.argv)

    result_window = ResultWindow('seg', './shell/train_seg_temp.sh')
    result_window.show()
    result_window.start_script()

    sys.exit(app.exec_())