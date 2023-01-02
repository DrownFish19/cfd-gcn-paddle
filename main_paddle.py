import os
import sys
from argparse import ArgumentParser

import numpy as np

os.environ['SU2_RUN'] = '/root/autodl-tmp/SU2_bin'
sys.path.append('/root/autodl-tmp/SU2_bin')

import paddle
# from paddle.io import DataLoader
from pgl.utils.data.dataloader import Dataloader
from paddle import nn, optimizer
from common import make_grid
from paddle import to_tensor

from mesh_utils import plot_field, is_ccw
# from su2paddle import activate_su2_mpi
from data import MeshAirfoilDataset
from models import CFDGCN, MeshGCN, UCM, CFD


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp-name', '-e', default='gcn_interp',
                        help='Experiment name, defaults to model name.')
    parser.add_argument('--su2-config', '-sc', default='coarse.cfg')
    parser.add_argument('--data-dir', '-d', default='data/NACA0012_interpolate',
                        help='Directory with dataset.')
    parser.add_argument('--coarse-mesh', default='meshes/mesh_NACA0012_xcoarse.su2',
                        help='Path to coarse mesh (required for CFD-GCN).')
    parser.add_argument('--version', type=int, default=None,
                        help='If specified log version doesnt exist, create it.'
                             ' If it exists, continue from where it stopped.')
    parser.add_argument('--load-model', '-lm', default='', help='Load previously trained model.')

    parser.add_argument('--model', '-m', default='gcn',
                        help='Which model to use.')
    parser.add_argument('--max-epochs', '-me', type=int, default=1000,
                        help='Max number of epochs to train for.')
    parser.add_argument('--optim', default='adam', help='Optimizer.')
    parser.add_argument('--batch-size', '-bs', type=int, default=4)
    parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, default=5e-5)
    parser.add_argument('--num-layers', '-nl', type=int, default=3)
    parser.add_argument('--num-end-convs', type=int, default=3)
    parser.add_argument('--hidden-size', '-hs', type=int, default=512)
    parser.add_argument('--freeze-mesh', action='store_true',
                        help='Do not do any learning on the mesh.')

    parser.add_argument('--eval', action='store_true',
                        help='Skips training, does only eval.')
    parser.add_argument('--profile', action='store_true',
                        help='Run profiler.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus to use, 0 for none.')
    parser.add_argument('--dataloader-workers', '-dw', type=int, default=2,
                        help='Number of Pytorch Dataloader workers to use.')
    parser.add_argument('--train-val-split', '-tvs', type=float, default=0.9,
                        help='Percentage of training set to use for training.')
    parser.add_argument('--val-check-interval', '-vci', type=int, default=None,
                        help='Run validation every N batches, '
                             'defaults to once every epoch.')
    parser.add_argument('--early-stop-patience', '-esp', type=int, default=0,
                        help='Patience before early stopping. '
                             'Does not early stop by default.')
    parser.add_argument('--train-pct', type=float, default=1.0,
                        help='Run on a reduced percentage of the training set,'
                             ' defaults to running with full data.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1],
                        help='Verbosity level. Defaults to 1, 0 for quiet.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode. Doesnt write logs. Runs '
                             'a single iteration of training and validation.')
    parser.add_argument('--no-log', action='store_true',
                        help='Dont save any logs or checkpoints.')

    args = parser.parse_args()
    args.nodename = os.uname().nodename
    if args.exp_name == '':
        args.exp_name = args.model
    if args.val_check_interval is None:
        args.val_check_interval = 1.0
    args.distributed_backend = 'dp'

    return args


def collate_fn(batch_data):
    # graphs = []
    # labels = []
    # for g in batch_data:
    #     graphs.append(g)
    #     labels.append(g.y)
    #
    # return graphs, labels

    return batch_data


class PaddleWrapper:
    def __init__(self, hparams):
        self.hparams = hparams
        self.step = None  # count test step because apparently Trainer doesnt
        self.criterion = nn.MSELoss()
        self.data = MeshAirfoilDataset(hparams.data_dir, mode='train')
        self.val_data = MeshAirfoilDataset(hparams.data_dir, mode='test')
        self.test_data = MeshAirfoilDataset(hparams.data_dir, mode='test')

        in_channels = self.data[0].node_feat["feature"].shape[-1]
        out_channels = self.data[0].y.shape[-1]
        hidden_channels = hparams.hidden_size

        if hparams.model == 'cfd_gcn':
            self.model = CFDGCN(hparams.su2_config,
                                self.hparams.coarse_mesh,
                                fine_marker_dict=self.data.marker_dict,
                                hidden_channels=hidden_channels,
                                num_convs=self.hparams.num_layers,
                                num_end_convs=self.hparams.num_end_convs,
                                out_channels=out_channels,
                                process_sim=self.data.preprocess,
                                freeze_mesh=self.hparams.freeze_mesh,
                                device='cuda' if self.hparams.gpus > 0 else 'cpu')
        elif hparams.model == 'gcn':
            self.model = MeshGCN(in_channels,
                                 hidden_channels,
                                 out_channels,
                                 fine_marker_dict=self.data.marker_dict,
                                 num_layers=hparams.num_layers)
        elif hparams.model == 'ucm':
            self.model = UCM(hparams.su2_config,
                             self.hparams.coarse_mesh,
                             fine_marker_dict=self.data.marker_dict,
                             process_sim=self.data.preprocess,
                             freeze_mesh=self.hparams.freeze_mesh,
                             device='cuda' if self.hparams.gpus > 0 else 'cpu')
        elif hparams.model == 'cfd':
            self.model = CFD(hparams.su2_config,
                             self.hparams.coarse_mesh,
                             fine_marker_dict=self.data.marker_dict,
                             process_sim=self.data.preprocess,
                             freeze_mesh=self.hparams.freeze_mesh,
                             device='cuda' if self.hparams.gpus > 0 else 'cpu')
        else:
            raise NotImplementedError

        # config optimizer
        self.parameters = self.model.parameters()
        if self.hparams.optim.lower() == 'adam':
            self.optimizer = optimizer.Adam(parameters=self.parameters, learning_rate=self.hparams.lr)
        elif self.hparams.optim.lower() == 'rmsprop':
            self.optimizer = optimizer.RMSProp(parameters=self.parameters, learning_rate=self.hparams.lr)
        elif self.hparams.optim.lower() == 'sgd':
            self.optimizer = optimizer.SGD(parameters=self.parameters, learning_rate=self.hparams.lr)
        else:
            self.optimizer = optimizer.SGD(parameters=self.parameters, learning_rate=self.hparams.lr)

        # config dataloader
        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.test_loader = self.test_dataloader()

        # config criterion
        self.criterion = paddle.nn.loss.MSELoss()

        self.sum_loss = 0.0
        self.global_step = 0

    def on_epoch_start(self):
        print('------', flush=True)
        self.sum_loss = 0.0

    def on_epoch_end(self):
        avg_loss = self.sum_loss / max(len(self.train_loader), 1)
        print("train_loss:{},step:{}".format(avg_loss, self.global_step), flush=True)

    def common_step(self, graphs):
        loss = 0.0
        pred_fields = []
        for graph in graphs:
            true_field = graph.y
            pred_field = self.model(graph, paddle.to_tensor(graph.node_feat["feature"]))
            mse_loss = self.criterion(pred_field, true_field)
            loss += mse_loss

        loss = loss / len(graphs)
        # pred_fields = paddle.stack(pred_fields)
        self.global_step += 1

        return loss, pred_fields

    def training_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)

        # if batch_idx + 1 == self.trainer.val_check_batch:
        #     # log images when doing val check
        #     self.log_images(batch.x[:, :2], pred, batch.y, batch.batch,
        #                     self.data.elems_list, 'train')

        self.sum_loss += loss.item()

        print("batch_train_loss:{}".format(loss.item()), flush=True)

        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

    def validation_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)

        # if batch_idx == 0:
            # log images only once per val epoch
            # self.log_images(batch.x[:, :2], pred, batch.y, batch.batch, self.data.elems_list, 'val')

        print("batch_val_loss:{}".format(loss.item()), flush=True)

        return loss.item()

    def test_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)

        # batch_size = batch.batch.max()
        self.step = 0 if self.step is None else self.step
        # for i in range(batch_size):
        #     self.log_images(batch.x[:, :2], pred, batch.y, batch.batch, self.data.elems_list, 'test', log_idx=i)
        self.step += 1

        return loss.item()

    def train_dataloader(self):
        train_loader = Dataloader(self.data,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=(self.hparams.train_pct == 1.0),  # don't shuffle if using reduced set
                                  num_workers=1,
                                  collate_fn=collate_fn)
        if self.hparams.verbose:
            print(f'Train data: {len(self.data)} examples, 'f'{len(train_loader)} batches.', flush=True)
        return train_loader

    def val_dataloader(self):
        # use test data here to get full training curve for test set
        val_loader = Dataloader(self.val_data,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=collate_fn)
        if self.hparams.verbose:
            print(f'Val data: {len(self.val_data)} examples, 'f'{len(val_loader)} batches.', flush=True)
        return val_loader

    def test_dataloader(self):
        test_loader = Dataloader(self.test_data,
                                 batch_size=self.hparams.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=collate_fn)
        if self.hparams.verbose:
            print(f'Test data: {len(self.test_data)} examples, 'f'{len(test_loader)} batches.', flush=True)
        return test_loader

    def log_images(self, nodes, pred, true, batch, elems_list, mode, log_idx=0):
        # if self.hparams.no_log or self.logger.debug:
        #     return

        inds = batch == log_idx
        nodes = nodes[inds]
        pred = pred[inds]
        true = true[inds]

        # exp = self.logger.experiment
        # step = self.trainer.global_step if self.step is None else self.step
        for field in range(pred.shape[1]):
            true_img = plot_field(nodes, elems_list, true[:, field],
                                  title='true')
            true_img = to_tensor(true_img)
            min_max = (true[:, field].min().item(), true[:, field].max().item())
            pred_img = plot_field(nodes, elems_list, pred[:, field],
                                  title='pred', clim=min_max)
            pred_img = to_tensor(pred_img)
            imgs = [pred_img, true_img]
            if hasattr(self.model, 'sim_info'):
                sim = self.model.sim_info
                sim_inds = sim['batch'] == log_idx
                sim_nodes = sim['nodes'][sim_inds]
                sim_info = sim['output'][sim_inds]
                sim_elems = sim['elems'][log_idx]

                mesh_inds = paddle.full_like(sim['batch'], fill_value=-1, dtype=paddle.int64)
                mesh_inds[sim_inds] = paddle.arange(sim_nodes.shape[0])
                sim_elems_list = self.model.contiguous_elems_list(sim_elems, mesh_inds)

                sim_img = plot_field(sim_nodes, sim_elems_list, sim_info[:, field],
                                     title='sim', clim=min_max)
                sim_img = to_tensor(sim_img)
                imgs = [sim_img] + imgs

            grid = make_grid(paddle.stack(imgs), padding=0)
            img_name = f'{mode}_pred_f{field}'
            # exp.add_image(img_name, grid, global_step=step)

    @staticmethod
    def get_cross_prods(meshes, store_elems):
        cross_prods = [is_ccw(mesh[e, :2], ret_val=True)
                       for mesh, elems in zip(meshes, store_elems) for e in elems]
        return cross_prods


if __name__ == '__main__':
    paddle.set_device("gpu")
    # activate_su2_mpi(remove_temp_files=True)

    args = parse_args()
    print(args, file=sys.stderr)
    paddle.seed(args.seed)

    trainer = PaddleWrapper(args)

    for epoch in range(args.max_epochs):
        trainer.on_epoch_start()

        # for train
        for i, graphs in enumerate(trainer.train_loader()):
            trainer.training_step(graphs, i)

        trainer.on_epoch_end()

        # for val
        total_val_loss = []
        for i, x in enumerate(trainer.val_loader):
            val_loss = trainer.validation_step(x, i)
            total_val_loss.append(val_loss)
        mean_val_loss = np.stack(total_val_loss).mean()
        print("val_loss (mean):{}".format(mean_val_loss), flush=True)

        # for test
        total_test_loss = []
        for i, x in enumerate(trainer.test_loader):
            test_loss = trainer.test_step(x, i)
            total_test_loss.append(test_loss)
        mean_test_loss = np.stack(total_test_loss).mean()
        print("test_loss (mean):{}".format(mean_test_loss), flush=True)


