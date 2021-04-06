# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1347, 910)
        self.load_model = QtWidgets.QPushButton(Dialog)
        self.load_model.setGeometry(QtCore.QRect(10, 20, 101, 31))
        self.load_model.setObjectName("load_model")
        self.load_weight = QtWidgets.QPushButton(Dialog)
        self.load_weight.setGeometry(QtCore.QRect(130, 20, 101, 31))
        self.load_weight.setObjectName("load_weight")
        self.load_dataset = QtWidgets.QPushButton(Dialog)
        self.load_dataset.setGeometry(QtCore.QRect(250, 20, 101, 31))
        self.load_dataset.setObjectName("load_dataset")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.load_model.setText(_translate("Dialog", "Load Model"))
        self.load_weight.setText(_translate("Dialog", "Load Weight"))
        self.load_dataset.setText(_translate("Dialog", "Load Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

