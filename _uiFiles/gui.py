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
        Dialog.resize(1300, 955)
        self.tab = QtWidgets.QTabWidget(Dialog)
        self.tab.setGeometry(QtCore.QRect(0, 0, 1191, 881))
        self.tab.setObjectName("tab")
        self.main = QtWidgets.QWidget()
        self.main.setObjectName("main")
        self.map_data = QtWidgets.QLineEdit(self.main)
        self.map_data.setGeometry(QtCore.QRect(50, 70, 71, 25))
        self.map_data.setText("")
        self.map_data.setAlignment(QtCore.Qt.AlignCenter)
        self.map_data.setObjectName("map_data")
        self.ade6 = QtWidgets.QLineEdit(self.main)
        self.ade6.setGeometry(QtCore.QRect(660, 70, 41, 25))
        self.ade6.setAlignment(QtCore.Qt.AlignCenter)
        self.ade6.setObjectName("ade6")
        self.prev_idx = QtWidgets.QPushButton(self.main)
        self.prev_idx.setGeometry(QtCore.QRect(1030, 70, 61, 25))
        self.prev_idx.setObjectName("prev_idx")
        self.num_of_vehicles_data = QtWidgets.QLineEdit(self.main)
        self.num_of_vehicles_data.setGeometry(QtCore.QRect(230, 70, 41, 25))
        self.num_of_vehicles_data.setText("")
        self.num_of_vehicles_data.setAlignment(QtCore.Qt.AlignCenter)
        self.num_of_vehicles_data.setObjectName("num_of_vehicles_data")
        self.ade6_data = QtWidgets.QLineEdit(self.main)
        self.ade6_data.setGeometry(QtCore.QRect(700, 70, 61, 25))
        self.ade6_data.setText("")
        self.ade6_data.setAlignment(QtCore.Qt.AlignCenter)
        self.ade6_data.setObjectName("ade6_data")
        self.SceneInfo_data = QtWidgets.QLineEdit(self.main)
        self.SceneInfo_data.setGeometry(QtCore.QRect(360, 70, 71, 25))
        self.SceneInfo_data.setText("")
        self.SceneInfo_data.setAlignment(QtCore.Qt.AlignCenter)
        self.SceneInfo_data.setObjectName("SceneInfo_data")
        self.modelInfo = QtWidgets.QLineEdit(self.main)
        self.modelInfo.setGeometry(QtCore.QRect(10, 10, 1001, 25))
        self.modelInfo.setObjectName("modelInfo")
        self.loadModel = QtWidgets.QPushButton(self.main)
        self.loadModel.setGeometry(QtCore.QRect(1030, 10, 141, 25))
        self.loadModel.setObjectName("loadModel")
        self.fde6 = QtWidgets.QLineEdit(self.main)
        self.fde6.setGeometry(QtCore.QRect(770, 70, 41, 25))
        self.fde6.setAlignment(QtCore.Qt.AlignCenter)
        self.fde6.setObjectName("fde6")
        self.map = QtWidgets.QLineEdit(self.main)
        self.map.setGeometry(QtCore.QRect(10, 70, 41, 25))
        self.map.setAlignment(QtCore.Qt.AlignCenter)
        self.map.setObjectName("map")
        self.dataInfo = QtWidgets.QLineEdit(self.main)
        self.dataInfo.setGeometry(QtCore.QRect(10, 40, 1001, 25))
        self.dataInfo.setObjectName("dataInfo")
        self.export_2 = QtWidgets.QPushButton(self.main)
        self.export_2.setGeometry(QtCore.QRect(1030, 800, 141, 25))
        self.export_2.setObjectName("export_2")
        self.num_of_vehicles = QtWidgets.QLineEdit(self.main)
        self.num_of_vehicles.setGeometry(QtCore.QRect(130, 70, 101, 25))
        self.num_of_vehicles.setAlignment(QtCore.Qt.AlignCenter)
        self.num_of_vehicles.setObjectName("num_of_vehicles")
        self.ade1_data = QtWidgets.QLineEdit(self.main)
        self.ade1_data.setGeometry(QtCore.QRect(480, 70, 61, 25))
        self.ade1_data.setText("")
        self.ade1_data.setAlignment(QtCore.Qt.AlignCenter)
        self.ade1_data.setObjectName("ade1_data")
        self.fde6_data = QtWidgets.QLineEdit(self.main)
        self.fde6_data.setGeometry(QtCore.QRect(810, 70, 61, 25))
        self.fde6_data.setText("")
        self.fde6_data.setAlignment(QtCore.Qt.AlignCenter)
        self.fde6_data.setObjectName("fde6_data")
        self.fde1_data = QtWidgets.QLineEdit(self.main)
        self.fde1_data.setGeometry(QtCore.QRect(590, 70, 61, 25))
        self.fde1_data.setText("")
        self.fde1_data.setAlignment(QtCore.Qt.AlignCenter)
        self.fde1_data.setObjectName("fde1_data")
        self.idx_data = QtWidgets.QLineEdit(self.main)
        self.idx_data.setGeometry(QtCore.QRect(910, 70, 101, 25))
        self.idx_data.setText("")
        self.idx_data.setAlignment(QtCore.Qt.AlignCenter)
        self.idx_data.setObjectName("idx_data")
        self.fde1 = QtWidgets.QLineEdit(self.main)
        self.fde1.setGeometry(QtCore.QRect(550, 70, 41, 25))
        self.fde1.setAlignment(QtCore.Qt.AlignCenter)
        self.fde1.setObjectName("fde1")
        self.ade1 = QtWidgets.QLineEdit(self.main)
        self.ade1.setGeometry(QtCore.QRect(440, 70, 41, 25))
        self.ade1.setAlignment(QtCore.Qt.AlignCenter)
        self.ade1.setObjectName("ade1")
        self.show_predict = QtWidgets.QCheckBox(self.main)
        self.show_predict.setGeometry(QtCore.QRect(1030, 440, 136, 32))
        self.show_predict.setObjectName("show_predict")
        self.next_idx = QtWidgets.QPushButton(self.main)
        self.next_idx.setGeometry(QtCore.QRect(1110, 70, 61, 25))
        self.next_idx.setObjectName("next_idx")
        self.pred_plot = matplotlibWidget(self.main)
        self.pred_plot.setGeometry(QtCore.QRect(10, 110, 1001, 721))
        self.pred_plot.setObjectName("pred_plot")
        self.splitter = QtWidgets.QSplitter(self.main)
        self.splitter.setGeometry(QtCore.QRect(1030, 110, 136, 321))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setOpaqueResize(False)
        self.splitter.setHandleWidth(4)
        self.splitter.setObjectName("splitter")
        self.ego_path_enable = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_enable.setObjectName("ego_path_enable")
        self.ego_path_aug_1 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_1.setObjectName("ego_path_aug_1")
        self.ego_path_aug_2 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_2.setObjectName("ego_path_aug_2")
        self.ego_path_aug_3 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_3.setObjectName("ego_path_aug_3")
        self.ego_path_aug_4 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_4.setObjectName("ego_path_aug_4")
        self.ego_path_aug_5 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_5.setObjectName("ego_path_aug_5")
        self.ego_path_aug_6 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_6.setObjectName("ego_path_aug_6")
        self.ego_path_aug_7 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_7.setObjectName("ego_path_aug_7")
        self.ego_path_aug_8 = QtWidgets.QCheckBox(self.splitter)
        self.ego_path_aug_8.setObjectName("ego_path_aug_8")
        self.SceneInfo = QtWidgets.QLineEdit(self.main)
        self.SceneInfo.setGeometry(QtCore.QRect(280, 70, 81, 25))
        self.SceneInfo.setAlignment(QtCore.Qt.AlignCenter)
        self.SceneInfo.setObjectName("SceneInfo")
        self.idx = QtWidgets.QLineEdit(self.main)
        self.idx.setGeometry(QtCore.QRect(880, 70, 31, 25))
        self.idx.setAlignment(QtCore.Qt.AlignCenter)
        self.idx.setObjectName("idx")
        self.loadData = QtWidgets.QPushButton(self.main)
        self.loadData.setGeometry(QtCore.QRect(1030, 40, 141, 25))
        self.loadData.setObjectName("loadData")
        self.tab.addTab(self.main, "")
        self.table = QtWidgets.QWidget()
        self.table.setObjectName("table")
        self.tableWidget = QtWidgets.QTableWidget(self.table)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 961, 841))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tableWidget.setFont(font)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(8)
        self.tableWidget.setRowCount(30)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(16, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(17, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(18, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(19, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(20, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(21, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(22, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(23, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(24, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(25, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(26, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(27, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(28, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(29, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(7, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(97)
        self.tableWidget.verticalHeader().setDefaultSectionSize(25)
        self.tab.addTab(self.table, "")

        self.retranslateUi(Dialog)
        self.tab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.ade6.setText(_translate("Dialog", "ade6"))
        self.prev_idx.setText(_translate("Dialog", "prev"))
        self.loadModel.setText(_translate("Dialog", "Load Model"))
        self.fde6.setText(_translate("Dialog", "fde6"))
        self.map.setText(_translate("Dialog", "MAP"))
        self.export_2.setText(_translate("Dialog", "export"))
        self.num_of_vehicles.setText(_translate("Dialog", "# of vehicles"))
        self.fde1.setText(_translate("Dialog", "fde1"))
        self.ade1.setText(_translate("Dialog", "ade1"))
        self.show_predict.setText(_translate("Dialog", "plot prediction"))
        self.next_idx.setText(_translate("Dialog", "next"))
        self.ego_path_enable.setText(_translate("Dialog", "ego path in data"))
        self.ego_path_aug_1.setText(_translate("Dialog", "ego path cand 1"))
        self.ego_path_aug_2.setText(_translate("Dialog", "ego path cand 2"))
        self.ego_path_aug_3.setText(_translate("Dialog", "ego path cand 3"))
        self.ego_path_aug_4.setText(_translate("Dialog", "ego path cand 4"))
        self.ego_path_aug_5.setText(_translate("Dialog", "ego path cand 5"))
        self.ego_path_aug_6.setText(_translate("Dialog", "ego path cand 6"))
        self.ego_path_aug_7.setText(_translate("Dialog", "ego path cand 7"))
        self.ego_path_aug_8.setText(_translate("Dialog", "ego path cand 8"))
        self.SceneInfo.setText(_translate("Dialog", "Scene Info"))
        self.idx.setText(_translate("Dialog", "idx"))
        self.loadData.setText(_translate("Dialog", "Load Data"))
        self.tab.setTabText(self.tab.indexOf(self.main), _translate("Dialog", "Main"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Dialog", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("Dialog", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("Dialog", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("Dialog", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("Dialog", "5"))
        item = self.tableWidget.verticalHeaderItem(5)
        item.setText(_translate("Dialog", "6"))
        item = self.tableWidget.verticalHeaderItem(6)
        item.setText(_translate("Dialog", "7"))
        item = self.tableWidget.verticalHeaderItem(7)
        item.setText(_translate("Dialog", "8"))
        item = self.tableWidget.verticalHeaderItem(8)
        item.setText(_translate("Dialog", "9"))
        item = self.tableWidget.verticalHeaderItem(9)
        item.setText(_translate("Dialog", "10"))
        item = self.tableWidget.verticalHeaderItem(10)
        item.setText(_translate("Dialog", "11"))
        item = self.tableWidget.verticalHeaderItem(11)
        item.setText(_translate("Dialog", "12"))
        item = self.tableWidget.verticalHeaderItem(12)
        item.setText(_translate("Dialog", "13"))
        item = self.tableWidget.verticalHeaderItem(13)
        item.setText(_translate("Dialog", "14"))
        item = self.tableWidget.verticalHeaderItem(14)
        item.setText(_translate("Dialog", "15"))
        item = self.tableWidget.verticalHeaderItem(15)
        item.setText(_translate("Dialog", "16"))
        item = self.tableWidget.verticalHeaderItem(16)
        item.setText(_translate("Dialog", "17"))
        item = self.tableWidget.verticalHeaderItem(17)
        item.setText(_translate("Dialog", "18"))
        item = self.tableWidget.verticalHeaderItem(18)
        item.setText(_translate("Dialog", "19"))
        item = self.tableWidget.verticalHeaderItem(19)
        item.setText(_translate("Dialog", "20"))
        item = self.tableWidget.verticalHeaderItem(20)
        item.setText(_translate("Dialog", "21"))
        item = self.tableWidget.verticalHeaderItem(21)
        item.setText(_translate("Dialog", "22"))
        item = self.tableWidget.verticalHeaderItem(22)
        item.setText(_translate("Dialog", "23"))
        item = self.tableWidget.verticalHeaderItem(23)
        item.setText(_translate("Dialog", "24"))
        item = self.tableWidget.verticalHeaderItem(24)
        item.setText(_translate("Dialog", "25"))
        item = self.tableWidget.verticalHeaderItem(25)
        item.setText(_translate("Dialog", "26"))
        item = self.tableWidget.verticalHeaderItem(26)
        item.setText(_translate("Dialog", "27"))
        item = self.tableWidget.verticalHeaderItem(27)
        item.setText(_translate("Dialog", "28"))
        item = self.tableWidget.verticalHeaderItem(28)
        item.setText(_translate("Dialog", "29"))
        item = self.tableWidget.verticalHeaderItem(29)
        item.setText(_translate("Dialog", "30"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "indata_x"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "indata_y"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "aug1_x"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "aug1_y"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "aug2_x"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("Dialog", "aug2_y"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("Dialog", "aug3_x"))
        item = self.tableWidget.horizontalHeaderItem(7)
        item.setText(_translate("Dialog", "aug3_y"))
        self.tab.setTabText(self.tab.indexOf(self.table), _translate("Dialog", "table"))
from matplotlibwidgetFile import matplotlibWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
