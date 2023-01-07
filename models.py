import os

import paddle
import paddle.nn.functional as F
from paddle import nn
from pgl.nn import GCNConv

from mesh_utils import write_graph_mesh, quad2tri, get_mesh_graph, signed_dist_graph, is_cw
from su2paddle import SU2Module


# import torch_geometric
# from torch_geometric.nn.conv import GCNConv
# from torch_geometric.nn.unpool import knn_interpolate

# class GCNConv(nn.Layer):
#     def __init__(self, input_size, output_size, activation='relu'):
#         super(GCNConv, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.linear = nn.Linear(input_size, output_size)
#         self.activation = activation
#
#     def forward(self, x, edge_index):
#         batch_size = x.shape[0]
#         src_index = edge_index[:, 0, :].reshape([-1])
#         dst_index = edge_index[:, 1, :].reshape([-1])
#
#         x = self.linear(x.reshape([-1, x.shape[-1]]))
#         x = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="mean")
#         # x = nn.functional.relu(x)
#
#         return x.reshape([batch_size, -1, x.shape[-1]])


class MeshGCN(nn.Layer):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, fine_marker_dict=None):
        super().__init__()
        self.fine_marker_dict = paddle.unique(paddle.to_tensor(fine_marker_dict['airfoil']))
        self.sdf = None
        in_channels += 1  # account for sdf

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels += [out_channels]

        self.convs = nn.LayerList()
        for i in range(num_layers):
            self.convs.append(GCNConv(channels[i], channels[i + 1]))

        # self.convs.append(GATConv(channels[0], channels[1], num_heads=4))
        # for i in range(1, num_layers - 1):
        #     self.convs.append(GATConv(channels[i] * 4, channels[i + 1], num_heads=4))
        # self.convs.append(GATConv(channels[num_layers - 1]*4, channels[num_layers], num_heads=1))

    def forward(self, graph, x):

        if self.sdf is None:
            with paddle.no_grad():
                self.sdf = signed_dist_graph(x[:, :2], self.fine_marker_dict).unsqueeze(1)
        x = paddle.concat([x, self.sdf], axis=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(graph, x)
            x = F.relu(x)

        x = self.convs[-1](graph, x)
        return x


class CFDGCN(nn.Layer):
    def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512,
                 out_channels=3):
        super().__init__()
        meshes_temp_dir = 'temp_meshes'
        os.makedirs(meshes_temp_dir, exist_ok=True)
        self.mesh_file = meshes_temp_dir + '/' + str(os.getpid()) + '_mesh.su2'

        if not coarse_mesh:
            raise ValueError('Need to provide a coarse mesh for CFD-GCN.')
        nodes, edges, self.elems, self.marker_dict = get_mesh_graph(coarse_mesh)
        if not freeze_mesh:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=False)
        else:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=True)

        self.elems, new_edges = quad2tri(sum(self.elems, []))
        self.elems = [self.elems]
        self.edges = paddle.to_tensor(edges)
        print(self.edges.dtype, new_edges.dtype)
        self.edges = paddle.concat([self.edges, new_edges], axis=1)
        self.marker_inds = paddle.to_tensor(sum(self.marker_dict.values(), [])).unique()
        assert is_cw(self.nodes, paddle.to_tensor(self.elems[0])).nonzero().shape[0] == 0, 'Mesh has flipped elems'

        self.process_sim = process_sim
        self.su2 = SU2Module(config_file, mesh_file=self.mesh_file)
        print(f'Mesh filename: {self.mesh_file.format(batch_index="*")}', flush=True)

        self.fine_marker_dict = paddle.to_tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None

        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.LayerList()
            in_channels = out_channels + hidden_channels
            for i in range(self.num_convs - 1):
                self.convs.append(GCNConv(in_channels, hidden_channels))
                in_channels = hidden_channels
            self.convs.append(GCNConv(in_channels, out_channels))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 + 1  # one extra channel for sdf
            self.pre_convs = nn.LayerList()
            for i in range(self.num_pre_convs - 1):
                self.pre_convs.append(GCNConv(in_channels, hidden_channels))
                in_channels = hidden_channels
            self.pre_convs.append(GCNConv(in_channels, hidden_channels))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, graph, x):
        batch_size = graph.aoa.shape[0]

        if self.sdf is None:
            with paddle.no_grad():
                self.sdf = signed_dist_graph(x[:, :2], self.fine_marker_dict).unsqueeze(1)
        fine_x = paddle.concat([x, self.sdf], axis=1)

        for i, conv in enumerate(self.pre_convs):
            fine_x = F.relu(conv(graph, fine_x))

        nodes = self.get_nodes()  # [353,2]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        # print("nodes[None, ..., 0]", nodes[None, ..., 0].shape, flush=True)
        # print("nodes[None, ..., 1]", nodes[None, ..., 1].shape, flush=True)
        # print("graph.aoa[..., None]", graph.aoa[..., None].shape, flush=True)
        # print("graph.mach_or_reynolds[..., None]", graph.mach_or_reynolds[..., None].shape,
        #       flush=True)
        batch_y = self.su2(nodes[None, ..., 0], nodes[None, ..., 1],
                           graph.aoa[..., None], graph.mach_or_reynolds[..., None])
        batch_y = self.process_sim(batch_y, False)

        coarse_y = paddle.stack([y.flatten() for y in batch_y], axis=1).astype("float32")  # features
        # print("coarse_y", coarse_y, flush=True)
        # print("coarse_y.shape", coarse_y.shape, flush=True)
        # print("nodes[:, :2]", nodes[:, :2], flush=True)
        # print("x[:, :2]", x[:, :2], flush=True)

        fine_y = self.upsample(features=coarse_y, coarse_nodes=nodes[:, :2], fine_nodes=x[:, :2])
        fine_y = paddle.concat([fine_y, fine_x], axis=1)

        for i, conv in enumerate(self.convs[:-1]):
            fine_y = F.relu(conv(graph, fine_y))
        fine_y = self.convs[-1](graph, fine_y)

        self.sim_info['nodes'] = nodes[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = graph
        self.sim_info['output'] = coarse_y

        return fine_y

    def upsample(self, features, coarse_nodes, fine_nodes):
        """

        :param features: [353，3]
        :param coarse_nodes: [353, 2]
        :param fine_nodes: [6684, 2]
        :return:
        """
        coarse_nodes_input = paddle.repeat_interleave(coarse_nodes.unsqueeze(0), fine_nodes.shape[0], 0)  # [6684,352,2]
        fine_nodes_input = paddle.repeat_interleave(fine_nodes.unsqueeze(1), coarse_nodes.shape[0], 1)  # [6684,352,2]

        dist_w = 1.0 / (paddle.norm(x=coarse_nodes_input - fine_nodes_input, p=2, axis=-1) + 1e-9)  # [6684,352]
        knn_value, knn_index = paddle.topk(dist_w, k=3, largest=True)  # [6684,3],[6684,3]

        weight = knn_value.unsqueeze(-2)
        features_input = features[knn_index]

        output = paddle.bmm(weight, features_input).squeeze(-2) / paddle.sum(knn_value, axis=-1, keepdim=True)

        # y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
        #                     coarse_batch.cpu(), fine.batch.cpu(), k=3)
        return output

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems


class UCM(CFDGCN):
    """Simply upsamples the coarse simulation without using any GCNs."""

    def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, coarse_mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = paddle.stack([batch.aoa, batch.mach_or_reynolds], axis=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        batch_x = batch_x.to('cpu', non_blocking=True)
        batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
                           batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        batch_y = [y.to(batch.x.device) for y in batch_y]
        batch_y = self.process_sim(batch_y, False)

        coarse_y = paddle.stack([y.flatten() for y in batch_y], axis=1)
        coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        zeros = batch.batch.new_zeros(num_nodes)
        coarse_batch = paddle.concat([zeros + i for i in range(batch_size)])

        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch)

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y


class CFD(CFDGCN):
    """Simply outputs the results of the (fine) CFD simulation."""

    def __init__(self, config_file, mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = paddle.stack([batch.aoa, batch.mach_or_reynolds], axis=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        batch_x = batch_x.to('cpu', non_blocking=True)
        batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
                           batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        batch_y = [y.to(batch.x.device) for y in batch_y]
        batch_y = self.process_sim(batch_y, False)

        coarse_y = paddle.stack([y.flatten() for y in batch_y], axis=1)
        coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        zeros = batch.batch.new_zeros(num_nodes)
        coarse_batch = paddle.concat([zeros + i for i in range(batch_size)])

        fine_y = coarse_y

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y
