
import sys
import ui.gui as ui


# MyDiag 모듈 안의 Ui_MyDialog 클래스로부터 파생
class XDialog(QDialog, ui.Ui_MyDialog):
    def __init__(self):
        QDialog.__init__(self)
        # setupUi() 메서드는 화면에 다이얼로그 보여줌
        self.setupUi(self)


app = QApplication(sys.argv)
dlg = XDialog()
dlg.show()
app.exec_()