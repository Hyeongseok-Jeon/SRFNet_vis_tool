# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1203, 861)
        self.loadModel = QtWidgets.QPushButton(Dialog)
        self.loadModel.setGeometry(QtCore.QRect(1040, 20, 141, 25))
        self.loadModel.setObjectName("loadModel")
        self.loadData = QtWidgets.QPushButton(Dialog)
        self.loadData.setGeometry(QtCore.QRect(1040, 50, 141, 25))
        self.loadData.setObjectName("loadData")
        self.modelInfo = QtWidgets.QLineEdit(Dialog)
        self.modelInfo.setGeometry(QtCore.QRect(20, 20, 1001, 25))
        self.modelInfo.setObjectName("modelInfo")
        self.dataInfo = QtWidgets.QLineEdit(Dialog)
        self.dataInfo.setGeometry(QtCore.QRect(20, 50, 1001, 25))
        self.dataInfo.setObjectName("dataInfo")
        self.map = QtWidgets.QLineEdit(Dialog)
        self.map.setGeometry(QtCore.QRect(20, 80, 41, 25))
        self.map.setAlignment(QtCore.Qt.AlignCenter)
        self.map.setObjectName("map")
        self.num_of_vehicles = QtWidgets.QLineEdit(Dialog)
        self.num_of_vehicles.setGeometry(QtCore.QRect(140, 80, 101, 25))
        self.num_of_vehicles.setAlignment(QtCore.Qt.AlignCenter)
        self.num_of_vehicles.setObjectName("num_of_vehicles")
        self.SceneInfo = QtWidgets.QLineEdit(Dialog)
        self.SceneInfo.setGeometry(QtCore.QRect(290, 80, 81, 25))
        self.SceneInfo.setAlignment(QtCore.Qt.AlignCenter)
        self.SceneInfo.setObjectName("SceneInfo")
        self.idx = QtWidgets.QLineEdit(Dialog)
        self.idx.setGeometry(QtCore.QRect(890, 80, 31, 25))
        self.idx.setAlignment(QtCore.Qt.AlignCenter)
        self.idx.setObjectName("idx")
        self.map_data = QtWidgets.QLineEdit(Dialog)
        self.map_data.setGeometry(QtCore.QRect(60, 80, 71, 25))
        self.map_data.setText("")
        self.map_data.setAlignment(QtCore.Qt.AlignCenter)
        self.map_data.setObjectName("map_data")
        self.num_of_vehicles_data = QtWidgets.QLineEdit(Dialog)
        self.num_of_vehicles_data.setGeometry(QtCore.QRect(240, 80, 41, 25))
        self.num_of_vehicles_data.setText("")
        self.num_of_vehicles_data.setAlignment(QtCore.Qt.AlignCenter)
        self.num_of_vehicles_data.setObjectName("num_of_vehicles_data")
        self.SceneInfo_data = QtWidgets.QLineEdit(Dialog)
        self.SceneInfo_data.setGeometry(QtCore.QRect(370, 80, 71, 25))
        self.SceneInfo_data.setText("")
        self.SceneInfo_data.setAlignment(QtCore.Qt.AlignCenter)
        self.SceneInfo_data.setObjectName("SceneInfo_data")
        self.idx_data = QtWidgets.QLineEdit(Dialog)
        self.idx_data.setGeometry(QtCore.QRect(920, 80, 101, 25))
        self.idx_data.setText("")
        self.idx_data.setAlignment(QtCore.Qt.AlignCenter)
        self.idx_data.setObjectName("idx_data")
        self.prev_idx = QtWidgets.QPushButton(Dialog)
        self.prev_idx.setGeometry(QtCore.QRect(1040, 80, 61, 25))
        self.prev_idx.setObjectName("prev_idx")
        self.next_idx = QtWidgets.QPushButton(Dialog)
        self.next_idx.setGeometry(QtCore.QRect(1120, 80, 61, 25))
        self.next_idx.setObjectName("next_idx")
        self.ade1 = QtWidgets.QLineEdit(Dialog)
        self.ade1.setGeometry(QtCore.QRect(450, 80, 41, 25))
        self.ade1.setAlignment(QtCore.Qt.AlignCenter)
        self.ade1.setObjectName("ade1")
        self.ade1_data = QtWidgets.QLineEdit(Dialog)
        self.ade1_data.setGeometry(QtCore.QRect(490, 80, 61, 25))
        self.ade1_data.setText("")
        self.ade1_data.setAlignment(QtCore.Qt.AlignCenter)
        self.ade1_data.setObjectName("ade1_data")
        self.fde1_data = QtWidgets.QLineEdit(Dialog)
        self.fde1_data.setGeometry(QtCore.QRect(600, 80, 61, 25))
        self.fde1_data.setText("")
        self.fde1_data.setAlignment(QtCore.Qt.AlignCenter)
        self.fde1_data.setObjectName("fde1_data")
        self.fde1 = QtWidgets.QLineEdit(Dialog)
        self.fde1.setGeometry(QtCore.QRect(560, 80, 41, 25))
        self.fde1.setAlignment(QtCore.Qt.AlignCenter)
        self.fde1.setObjectName("fde1")
        self.ade6 = QtWidgets.QLineEdit(Dialog)
        self.ade6.setGeometry(QtCore.QRect(670, 80, 41, 25))
        self.ade6.setAlignment(QtCore.Qt.AlignCenter)
        self.ade6.setObjectName("ade6")
        self.ade6_data = QtWidgets.QLineEdit(Dialog)
        self.ade6_data.setGeometry(QtCore.QRect(710, 80, 61, 25))
        self.ade6_data.setText("")
        self.ade6_data.setAlignment(QtCore.Qt.AlignCenter)
        self.ade6_data.setObjectName("ade6_data")
        self.fde6_data = QtWidgets.QLineEdit(Dialog)
        self.fde6_data.setGeometry(QtCore.QRect(820, 80, 61, 25))
        self.fde6_data.setText("")
        self.fde6_data.setAlignment(QtCore.Qt.AlignCenter)
        self.fde6_data.setObjectName("fde6_data")
        self.fde6 = QtWidgets.QLineEdit(Dialog)
        self.fde6.setGeometry(QtCore.QRect(780, 80, 41, 25))
        self.fde6.setAlignment(QtCore.Qt.AlignCenter)
        self.fde6.setObjectName("fde6")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.loadModel.setText(_translate("Dialog", "Load Model"))
        self.loadData.setText(_translate("Dialog", "Load Data"))
        self.map.setText(_translate("Dialog", "MAP"))
        self.num_of_vehicles.setText(_translate("Dialog", "# of vehicles"))
        self.SceneInfo.setText(_translate("Dialog", "Scene Info"))
        self.idx.setText(_translate("Dialog", "idx"))
        self.prev_idx.setText(_translate("Dialog", "prev"))
        self.next_idx.setText(_translate("Dialog", "next"))
        self.ade1.setText(_translate("Dialog", "ade1"))
        self.fde1.setText(_translate("Dialog", "fde1"))
        self.ade6.setText(_translate("Dialog", "ade6"))
        self.fde6.setText(_translate("Dialog", "fde6"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
