import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

DESCRIPTIONS = {'seg': 'TBA',
                'ood': 'TBA',
                'test': 'TBA'}

CB_OPTIONS = {'seg': [{'name': 'Epochs', 'options': ['100 (Default)', '200', '10']},
                    {'name': 'Loss Lambda', 'options': ['0.01 (Default)', '0.05', '0.1']}],
            'test': [{'name': 'YOLO Threshold', 'options': ['0.05 (Default)', '0.1', '0.2']},
                    {'name': 'OOD Threshold', 'options': ['87 (Default)', '95', '99']}]}

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.seg_folder = None
        self.ood_folder = None
        self.test_folder = None
        self.cb_epoch = None
        self.cb_llmbd = None
        self.initUI()

    def initUI(self):
        tabs = QTabWidget()

        tabs.addTab(self.create_tab(task='seg'), 'Train (Segmentation)')
        tabs.addTab(self.create_tab(task='ood'), 'Train (OOD Classifier)')
        tabs.addTab(self.create_tab(task='test'), 'Test')

        self.setCentralWidget(tabs)
        
        self._createStatusBar()
        self.setWindowTitle('Unknown Object detection')
        self.setWindowIcon(QIcon('./ui_utils/pil_logo_window.png'))
        self.resize(400, 400)

        self.center()
        self.show()
    
    def _createStatusBar(self):
        self.statusBar = self.statusBar()
        self.sbText = QLabel('2023, Developed by PIL')
        self.sbIcon = QLabel()
        self.sbIcon.setPixmap(QPixmap('./ui_utils/pil_logo_status.jpg').scaled(48,14))
        self.statusBar.addPermanentWidget(self.sbText)
        self.statusBar.addPermanentWidget(self.sbIcon)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def editTrainSegShell(self, sh_path, find, replacement):
        print(os.getcwd())
        with open(sh_path) as f:
            s = f.read()
        s = s.replace(find, replacement)
        
        with open(sh_path, "w") as f:
            f.write(s)
    

    def folderSegButtonClicked(self):
        self.seg_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.segFolderText.setText(f"'{self.seg_folder}'")
        self.segFolderText.setStyleSheet("color: blue;")

    def folderOodButtonClicked(self):
        self.ood_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.oodFolderText.setText(f"'{self.ood_folder}'")
        self.oodFolderText.setStyleSheet("color: blue;")
    
    def folderTestButtonClicked(self):
        self.test_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.testFolderText.setText(f"'{self.test_folder}'")
        self.testFolderText.setStyleSheet("color: blue;")
    
    
    def startButton(self, task):
        btnAct = {'seg': self.startSegButtonClicked,
                  'ood': self.startOodButtonClicked,
                  'test': self.startTestButtonClicked}
        
        button = QPushButton("Start", self)
        button.clicked.connect(btnAct[task])
        return button
    
    def startSegButtonClicked(self):
        script_path = './shell/train_seg.sh'
        
        if self.seg_folder is not None:
            self.editTrainSegShell(script_path, 
                    "--source /home/leeyoonji/workspace/git/datasets/mastr1478/images",
                    f"--source {self.seg_folder}")
        
        if self.cb_epoch is not None:
            self.editTrainSegShell(script_path, 
                    "--epochs 100",
                    f"--epochs {self.cb_epoch}")
        
        if self.cb_llmbd is not None:
            self.editTrainSegShell(script_path, 
                    "--separation_loss cwsl", 
                    f"--separation_loss {self.cb_llmbd}")
        
        os.system(f". {script_path} &")

    def startOodButtonClicked(self):
        script_path = './shell/train_ood.sh'
        
        if self.train_folder is not None:
            self.editTrainSegShell(script_path, 
                    "--source /home/leeyoonji/workspace/git/datasets/mastr1478/images",
                    f"--source {self.train_folder}")
        
        os.system(f". {script_path}")
    
    def startTestButtonClicked(self):
        script_path = './shell/infer_whole.sh'
        
        if self.train_folder is not None:
            self.editTrainSegShell(script_path, 
                    "--source ./datasets/custom102/images",
                    f"--source {self.train_folder}")
        
        if self.cb_yths is not None:
            self.editTrainSegShell(script_path, 
                    "--conf-thres 0.05",
                    f"--conf-thres {self.cb_yths}")
        
        if self.cb_oths is not None:
            self.editTrainSegShell(script_path, 
                    "--ood-thres 87", 
                    f"--ood-thres {self.cb_oths}")
        
        os.system(f". {script_path}")
    
    def onActivatedEpoch(self, text):
        self.cb_epoch = text.replace(' (Default)', '')

    def onActivatedLossLmbd(self, text):
        self.cb_llmbd = text.replace(' (Default)', '')
        
    def onActivatedYoloThs(self, text):
        self.cb_yths = text.replace(' (Default)', '')
        
    def onActivatedOodThs(self, text):
        self.cb_oths = text.replace(' (Default)', '')


    def createDescriptionGroup(self, task):
        groupbox = QGroupBox('Description')

        layout = QFormLayout()
        text = QLabel(DESCRIPTIONS[task])
        layout.addWidget(text)
        
        groupbox.setLayout(layout)
        return groupbox
    
    def createDatasetGroup(self, task):
        text = 'Test' if task == 'test' else 'Train'
        groupbox = QGroupBox(f'{text} Dataset')
        layout = QFormLayout()
        
        if task == 'seg':
            folderOpenButton = QPushButton('Open Folder',self)
            folderOpenButton.clicked.connect(self.folderSegButtonClicked)
            layout.addWidget(folderOpenButton)
            self.segFolderText = QLabel('')
            layout.addWidget(self.segFolderText)
            
        elif task == 'ood':
            folderOpenButton = QPushButton('Open Folder',self)
            folderOpenButton.clicked.connect(self.folderOodButtonClicked)
            layout.addWidget(folderOpenButton)
            self.oodFolderText = QLabel('')
            layout.addWidget(self.oodFolderText)
            
        elif task == 'test':
            folderOpenButton = QPushButton('Open Folder',self)
            folderOpenButton.clicked.connect(self.folderTestButtonClicked)
            layout.addWidget(folderOpenButton)
            self.testFolderText = QLabel('')
            layout.addWidget(self.testFolderText)
    
        groupbox.setLayout(layout)
        return groupbox
    
    def createHyperparamGroup(self, task):
        groupbox = QGroupBox('Hyperparameter Settings')
        groupbox.setCheckable(True)
        groupbox.setChecked(False)
        layout = QFormLayout()
        
        cbAct = {'seg': {'Epochs': self.onActivatedEpoch, 'Loss Lambda': self.onActivatedLossLmbd},
                      'test': {'YOLO Threshold': self.onActivatedYoloThs, 'OOD Threshold': self.onActivatedOodThs}}
        
        for hpDict in CB_OPTIONS[task]:
            name = hpDict['name']
            cb = QComboBox(self)
            cb.addItems(hpDict['options'])
            cb.activated[str].connect(cbAct[task][name])
            layout.addRow(f"- {name}:", cb)
        
        groupbox.setLayout(layout)
        return groupbox


    def create_tab(self, task):
        widget = QWidget()
        vbox = QVBoxLayout()
        
        vbox.addStretch(1)
        vbox.addWidget(self.createDescriptionGroup(task))
        vbox.addStretch(1)
        vbox.addWidget(self.createDatasetGroup(task))
        vbox.addStretch(1)
        if task != 'ood':
            vbox.addWidget(self.createHyperparamGroup(task))
            vbox.addStretch(1)
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.startButton(task))
        hbox.addStretch(0)

        vbox.addLayout(hbox)
        vbox.addStretch(1)

        widget.setLayout(vbox)
        return widget

        
if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())