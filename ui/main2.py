import sys
import pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from tab import TabWidget
from utils import WINDOW_TITLE


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window')

        tab_widget = QTabWidget(self)
        self.setCentralWidget(tab_widget)

        for task in ['seg', 'ood', 'test']:
            tab = TabWidget(task)
            tab_widget.addTab(tab, WINDOW_TITLE[task])

        self.setCentralWidget(tab_widget)
        
        self._createStatusBar()
        self.setWindowTitle('Unknown Object detection')

        icon_path = sorted(pathlib.Path('.').glob('**/pil_logo_window.png'))
        self.setWindowIcon(QIcon(str(icon_path[0])))
        
        self.resize(400, 400)
        self.center()
        self.show()
    
    def _createStatusBar(self):
        self.statusBar = self.statusBar()
        self.sbText = QLabel('2023, Developed by PIL')
        self.sbIcon = QLabel()

        icon_path = sorted(pathlib.Path('.').glob('**/pil_logo_status.jpg'))
        self.sbIcon.setPixmap(QPixmap(str(icon_path[0])).scaled(48,14))

        self.statusBar.addPermanentWidget(self.sbText)
        self.statusBar.addPermanentWidget(self.sbIcon)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
