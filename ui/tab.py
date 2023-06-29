import os
from PyQt5.QtWidgets import *
from result import ResultWindow
from utils import SCRIPT_PATH, DESCRIPTIONS, CB_OPTIONS


class TabWidget(QWidget):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.script_path = SCRIPT_PATH[task]
        self.data_folder = None
        self.cb_epoch = None
        self.cb_llmbd = None
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        
        vbox.addStretch(1)
        vbox.addWidget(self.createDescriptionGroup())
        vbox.addStretch(1)
        vbox.addWidget(self.createDatasetGroup())
        vbox.addStretch(1)
        if self.task != 'ood':
            vbox.addWidget(self.createHyperparamGroup())
            vbox.addStretch(1)
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.startButton())
        hbox.addStretch(0)

        vbox.addLayout(hbox)
        vbox.addStretch(1)

        self.setLayout(vbox)

    def createDescriptionGroup(self):
        groupbox = QGroupBox('Description')

        layout = QFormLayout()
        text = QLabel(DESCRIPTIONS[self.task])
        layout.addWidget(text)
        
        groupbox.setLayout(layout)
        return groupbox
    
    def createDatasetGroup(self):
        text = 'Test' if self.task == 'test' else 'Train'
        groupbox = QGroupBox(f'{text} Dataset')
        layout = QFormLayout()
    
        folderOpenButton = QPushButton('Open Folder',self)
        folderOpenButton.clicked.connect(self.folderButtonClicked)
        layout.addWidget(folderOpenButton)
        self.folderText = QLabel('')
        layout.addWidget(self.folderText)
        
        groupbox.setLayout(layout)
        return groupbox
    
    def createHyperparamGroup(self):
        groupbox = QGroupBox('Hyperparameter Settings')
        groupbox.setCheckable(True)
        groupbox.setChecked(False)
        layout = QFormLayout()
        
        cbAct = {'seg': {'Epochs': self.onActivatedEpoch, 'Loss Lambda': self.onActivatedLossLmbd},
                      'test': {'YOLO Threshold': self.onActivatedYoloThs, 'OOD Threshold': self.onActivatedOodThs}}
        
        for hpDict in CB_OPTIONS[self.task]:
            name = hpDict['name']
            cb = QComboBox(self)
            cb.addItems(hpDict['options'])
            cb.activated[str].connect(cbAct[self.task][name])
            layout.addRow(f"- {name}:", cb)
        
        groupbox.setLayout(layout)
        return groupbox

    def folderButtonClicked(self):
        self.data_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.folderText.setText(f"'{self.data_folder}'")
        self.folderText.setStyleSheet("color: blue;")

    def startButton(self):
        button = QPushButton("Start", self)
        button.clicked.connect(self.startButtonClicked)
        return button
    
    def startButtonClicked(self):
        if self.data_folder is not None:
            self.editShell("--source /home/leeyoonji/workspace/git/datasets/mastr1478/images",
                            f"--source {self.data_folder}")
        
        if self.task == 'seg':
            if self.cb_epoch is not None:
                self.editShell("--epochs 100",
                                f"--epochs {self.cb_epoch}")
            
            if self.cb_llmbd is not None:
                self.editShell("--separation_loss cwsl", 
                                f"--separation_loss {self.cb_llmbd}")
            
        elif self.task == 'test':
            if self.cb_yths is not None:
                self.editShell("--conf-thres 0.05",
                                f"--conf-thres {self.cb_yths}")
            
            if self.cb_oths is not None:
                self.editShell("--ood-thres 87", 
                                f"--ood-thres {self.cb_oths}")
            
        self.result_window = ResultWindow(self.task, self.script_path)
        self.result_window.show()
        self.result_window.start_script()

    def editShell(self, find, replacement):
        with open(self.script_path) as f:
            s = f.read()
        s = s.replace(find, replacement)
        
        if not '_edit.sh' in self.script_path:
            self.script_path = self.script_path.replace('.sh', '_edit.sh')
        
        with open(self.script_path, "w") as f:
            f.write(s)
    
    def onActivatedEpoch(self, text):
        self.cb_epoch = text.replace(' (Default)', '')

    def onActivatedLossLmbd(self, text):
        self.cb_llmbd = text.replace(' (Default)', '')
        
    def onActivatedYoloThs(self, text):
        self.cb_yths = text.replace(' (Default)', '')
        
    def onActivatedOodThs(self, text):
        self.cb_oths = text.replace(' (Default)', '')