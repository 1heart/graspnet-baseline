import os
import numpy as np
import open3d

from graspnet_baseline.demo import get_net
from graspnet_baseline.demo import process_data
from graspnet_baseline.demo import get_grasps
from graspnet_baseline.demo import collision_detection

import graspnetAPI

from polygrasp.grasp_rpc import GraspServer


class GraspNet1BServer(GraspServer):
    def __init__(self, net_cfg):
        self.net_cfg = net_cfg
        self.net = get_net(self.net_cfg)

        # GraspNet-1Billion requires flipping before processing, and undoing the transformation
        self.flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    def _get_grasps(self, pcd: open3d.geometry.PointCloud) -> graspnetAPI.GraspGroup:
        pcd = pcd.transform(self.flip)
        points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)

        end_points, cloud = process_data(self.net_cfg, points, colors)
        grasp_group = get_grasps(self.net, end_points)

        # Remove any grasp candidates that have collisions.
        if self.net_cfg.collision_thresh > 0:
            grasp_group = collision_detection(
                grasp_group, np.array(cloud.points), cfgs=self.net_cfg
            )

        grasp_group.nms()
        grasp_group.sort_by_score()
        grasp_group.transform(self.flip)

        return grasp_group


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="../../data/graspnet/checkpoint-rs.tar", help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, default="doc/example_data", help='Data directory')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    parser.add_argument('--device', type=str, default="cuda")
    cfgs = parser.parse_args()

    if not os.path.isabs(cfgs.checkpoint_path):
        cfgs.checkpoint_path = os.path.join(os.path.dirname(__file__), cfgs.checkpoint_path)

    server = GraspNet1BServer(cfgs)
    server.start()
