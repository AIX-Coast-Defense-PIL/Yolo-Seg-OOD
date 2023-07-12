import glob
from PyQt5.QtWidgets import *
from result import ResultWindow
from utils import *


class TabWidget(QWidget):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.data_folder = None
        self.cb_epoch = None
        self.cb_llmbd = None
        self.cb_yths = None
        self.cb_oths = None
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
        if '/images' in self.data_folder:
            self.data_folder = os.path.join(self.data_folder, os.pardir)
        self.folderText.setText(f"'{self.data_folder}'")
        self.folderText.setStyleSheet("color: blue;")

    def startButton(self):
        button = QPushButton("Start", self)
        button.clicked.connect(self.startButtonClicked)
        return button
    
    def startButtonClicked(self):
        script_path = SCRIPT_PATH[self.task]

        if self.task == 'seg':
            script_path = self.editShellSeg(script_path)
            
        elif self.task == 'ood':
            script_path = self.editShellOod()

        elif self.task == 'test':
            script_path = self.editShellTest(script_path)
        
        self.result_window = ResultWindow(self.task, len(script_path))
        self.result_window.show()
        self.result_window.start_script(script_path)
    
    def editShellSeg(self, script_path):
        if self.data_folder is not None:
            dir_name = self.data_folder.split('/')[-1]
            script_path = editFile(script_path, "--source /home/leeyoonji/workspace/git/datasets/mastr1478/images",
                                    f"--source {self.data_folder}/images")
            script_path = editFile(script_path, "--model_name mastr1478_wodis",
                                    f"--model_name {dir_name}_wodis")
                                    
        if self.cb_epoch is not None:
            script_path = editFile(script_path, "--epochs 100", 
                                f"--epochs {self.cb_epoch}")
                                
        if self.cb_llmbd is not None:
            script_path = editFile(script_path, "--separation_loss cwsl", 
                                f"--separation_loss {self.cb_llmbd}")
        return [script_path]
    
    def editShellOod(self):
        if self.data_folder is not None:
            refine_preds_path = editFile("./shell/refine_yolo_preds.sh", "data_dir=./data_example",
                                    f"data_dir={self.data_folder}")
            
            train_cluster_path = editFile("./shell/train_ood_cluster.sh", "--add_train_data None",
                                    f"--add_train_data {self.data_folder}")
            
        return [refine_preds_path, train_cluster_path]
    
    def editShellTest(self, script_path):
        if self.data_folder is not None:
            dir_name = self.data_folder.split('/')[-1]
            script_path = editFile(script_path, "--source ./datasets/custom102/images",
                                    f"--source {self.data_folder}/images")
            script_path = editFile(script_path, "--name custom102_$timestamp",
                                    f"--name {dir_name}_$timestamp")
                                    
        if self.cb_yths is not None:
            script_path = editFile(script_path, "--conf-thres 0.05", 
                                f"--conf-thres {self.cb_yths}")

        if self.cb_oths is not None:
            script_path = editFile(script_path, "--ood-thres 18", 
                                f"--ood-thres {self.cb_oths}")
        
        new_threshold_path = "./ood/cache/threshold/kmeans_resnet50_seaships_*.json"
        new_threshold = glob.glob(new_threshold_path)
        if new_threshold:
            script_path = editFile(script_path, "--threshold_path ./ood/cache/threshold/kmeans_resnet50_seaships.json",
                                f"--threshold_path {new_threshold[0]}")
        
        new_cluster_path = "./ood/cache/cluster/kmeans_resnet50_seaships_*.pkl"
        new_cluster = glob.glob(new_cluster_path)
        if new_cluster:
            script_path = editFile(script_path, "--cluster_path ./ood/cache/cluster/kmeans_resnet50_seaships.pkl",
                                f"--cluster_path {new_cluster[0]}")
        
        return [script_path]

    def onActivatedEpoch(self, text):
        self.cb_epoch = text.replace(' (Default)', '')

    def onActivatedLossLmbd(self, text):
        self.cb_llmbd = text.replace(' (Default)', '')
        
    def onActivatedYoloThs(self, text):
        self.cb_yths = text.replace(' (Default)', '')
        
    def onActivatedOodThs(self, text):
        self.cb_oths = text.replace(' (Default)', '')