
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import importlib.util
import sys
import tkinter
import tkinter.filedialog
import os
import _uiFiles.gui
import torch
import argparse
import time
from torch.utils.data import DataLoader
import argoverse.evaluation.eval_forecasting as eval


class MainDialog(QMainWindow, _uiFiles.gui.Ui_Dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.root_dir = os.getcwd()
        self.loadModel.clicked.connect(self.model_load)
        self.loadData.clicked.connect(self.data_load)
        self.next_idx.clicked.connect(self.next)
        self.idx = 0

    def next(self):
        self.cur_data = next(iter(self.data_loader))
        target_traj = self.cur_data['gt_preds'][0][1]
        while torch.norm(target_traj[0,:]-target_traj[-1,:]) < 2:
            self.cur_data = next(iter(self.data_loader))
        self.update_data()

    def update_data(self):
        print('update')
        self.map_data.setText(self.cur_data['city'][0])
        self.num_of_vehicles_data.setText(str(self.cur_data['feats'][0].shape[0]))
        self.SceneInfo_data.setText('None')
        self.idx_data.setText(str(self.cur_data['idx'][0]))
        _, self.pred_out, _, _, _, _, _, _ = self.net(self.cur_data)
        ade1, fde1, ade6, fde6 = self.get_eval_data(self.pred_out)
        self.ade1_data.setText(str(ade1)[:5])
        self.fde1_data.setText(str(fde1)[:5])
        self.ade6_data.setText(str(ade6)[:5])
        self.fde6_data.setText(str(fde6)[:5])
        self.visualization()

    # def visualization(self):


    def get_eval_data(self, pred_out):
        gt_trajectory = self.cur_data['gt_preds'][0][1, :, :]
        forecasted_trajectory = []
        score = []
        ade = []
        fde = []
        for i in range(6):
            score_pred = pred_out['cls'][0][0, i].cpu().detach().numpy()
            pred_traj = pred_out['reg'][0][0, i, :, :].cpu().detach().numpy()
            forecasted_trajectory.append(pred_traj)
            score.append(score_pred)
            ade.append(eval.get_ade(pred_traj, gt_trajectory))
            fde.append(eval.get_fde(pred_traj, gt_trajectory))
        conf_idx = np.where(score == max(score))[0][0]
        ade1 = ade[conf_idx]
        fde1 = fde[conf_idx]
        ade6 = np.min(ade)
        fde6 = np.min(fde)

        return ade1, fde1, ade6, fde6

    def data_load(self):
        config = dict()
        config['preprocess_val'] = os.path.join(self.root_dir, '../SRFNet/SRFNet/dataset/preprocess_GAN/val_crs_dist6_angle90.p')
        config['preprocess'] = True

        self.data = self.dataset(config, config, train=False)
        self.data_loader = DataLoader(self.data,
                                      batch_size=1,
                                      num_workers=1,
                                      collate_fn=self.collate_fn,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      )

        self.dataInfo.setText(os.path.join(self.root_dir, '../SRFNet/SRFNet/dataset/preprocess_GAN/val_crs_dist6_angle90.p'))
        self.cur_data = next(iter(self.data_loader))
        self.update_data()

    def model_load(self):
        root = tkinter.Tk()
        root.withdraw()

        currdir = os.getcwd()
        weight_file_dir = tkinter.filedialog.askopenfilename(parent=root, initialdir=os.path.join(currdir, '../SRFNet/SRFNet/results/'))
        weight_file = torch.load(weight_file_dir)

        model_case = weight_file_dir[str.find(weight_file_dir, 'model') + 6:]
        model_case = model_case[:str.find(model_case, '/')]

        sys.path.extend([os.path.join(os.path.dirname(weight_file_dir), 'files')])

        spec = importlib.util.spec_from_file_location('get_model', os.path.join(os.path.dirname(weight_file_dir), 'files', 'model_' + model_case + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        args = self.get_args()
        _, dataset, collate_fn, net, _, _, _, _ = module.get_model(args)
        pretrained_dict = weight_file['state_dict']
        net.load_state_dict(pretrained_dict)

        self.modelInfo.setText(weight_file_dir + ' is loaded successfully')
        self.net = net
        self.dataset = dataset
        self.collate_fn = collate_fn

    def get_args(self):
        parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
        parser.add_argument(
            "-m", "--model", default="model_lanegcn_GAN", type=str, metavar="MODEL", help="model name"
        )
        parser.add_argument("--eval", action="store_true")
        parser.add_argument(
            "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
        )
        parser.add_argument(
            "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
        )
        parser.add_argument(
            "--case", default="vanilla_gan", type=str
        )
        parser.add_argument(
            "--transfer", default=['encoder'], type=list
        )
        parser.add_argument("--mode", default='client')
        parser.add_argument("--port", default=52162)

        return parser.parse_args()


app = QApplication(sys.argv)
main_dialog = MainDialog()
main_dialog.show()
app.exec_()


#
# root_dir = '/home/jhs/Desktop/SRFNet/LaneGCN/dataset/val/data/'
#
# from argoverse.map_representation.map_api import ArgoverseMap
# from demo_usage.visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
# am = ArgoverseMap()
# from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
# afl = ArgoverseForecastingLoader(root_dir)
# from argoverse.utils.mpl_plotting_utils import draw_lane_polygons
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
#
# from argoverse.visualization.visualize_sequences import viz_sequence
# seq_path = f"{root_dir}/2645.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)
# xmin = 500
# xmax = 700
# ymin = 500
# ymax = 700
# city_name = 'MIA'
# local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
#
#
# local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
# local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
#
# domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix, use_existing_files=use_existing_files, log_id=argoverse_data.current_log)
#
# draw_lane_polygons(ax, local_lane_polygons, color='r')
