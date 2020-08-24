"""
MIT License

Copyright (c) 2020 Daniel Ordoñez

This code is partially based on code from Matthew Matl's package urdfpy

Copyright (c) 2019 Matthew Matl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict

import torch
import numpy as np
from urdfpy import URDF


class CartesianSpaceLoss(torch.nn.Module):
    """
    A cartesian space loss metric based on the conversion of state-space variables (e.g. joint angles,
    displacements) to links cartesian space through forward kinematics of non-closed kinematic chains, using as backbone
    an URDF definition of the kinematic chain.

    """

    def __init__(self, urdf, loss_type='frobenius', relevant_links=None, return_pos_error=False):
        """
        :param urdf: Kinematic chain definition using the URDF convention, used for calculation of the forward
        kinematics. To date only revolut joints and open-ended kinematic chains are supported.
        :type urdf: URDF
        :param loss_type: String indicating the type of loss. If the loss `loss_type` is `frobenius` the returned distance
            scalar is the average Frobenius norm across all `relevant_links` homogenous transformation matrices. If loss_type
            is `rot_only` the scalar loss is the Deviation from the Identity Matrix, defined in Huynh, Du Q. "Metrics
            for 3D rotations: Comparison and analysis." Journal of Mathematical Imaging and Vision 35.2 (2009): 155-164;
            which is an SO(3) distance metric, considering only the rotation matrices in the relevant links homogenous
            transformation matrices
        :type loss_type: string
        :param relevant_links: List of names (str) of the links in the URDF used for loss calculation and gradient
        backpropagation
        :type relevant_links: list
        :param return_pos_error: Whether to return the loss value along with a link cartesian position average error
        metric useful for quantitative analysis on results but not for gradient backpropagation as a main loss function.
        :type return_pos_error: bool
        """
        super(CartesianSpaceLoss, self).__init__()

        if not isinstance(urdf, URDF):
            raise ValueError("Invalid urdf argument")

        if relevant_links is not None:
            link_names = [link.name for link in urdf.links]
            for relevant_link in relevant_links:
                if relevant_link not in link_names:
                    raise ValueError("Relevant link %s not present in provided `urdf`" % relevant_link)
            # print("Loss will consider the following links %s " % relevant_links)

        if loss_type not in ['rot_only', 'frobenius', 'rot_loc']:
            raise ValueError("Invalid loss `loss_type` use: 'rot_only', 'frobenius' or 'rot_loc'")

        self.urdf = urdf
        self.loss_type = loss_type
        self.relevant_links = relevant_links
        self.return_pos_error = return_pos_error

    def forward(self, pred_joint_states, real_joint_states):
        """
        Forward pass of the loss function using the predicted and true joint states to estimate a distance metric of the
        robot/system links through the use of forward kinematics.
        :param pred_joint_states: Predicted joint angle values of all actuated joints
        :loss_type pred_joint_states: dict
        :param real_joint_states: Ground truth joint angle values of all actuated joints
        :loss_type real_joint_states: dict
        :return: A distance scalar metric in cartesian space. If the loss `loss_type` is `frobenius` the returned distance
            scalar is the average Frobenius norm across all `relevant_links` homogenous transformation matrices. If loss_type
            is `rot_only` the scalar loss is the Deviation from the Identity Matrix, defined in Huynh, Du Q. "Metrics
            for 3D rotations: Comparison and analysis." Journal of Mathematical Imaging and Vision 35.2 (2009): 155-164;
            which is an SO(3) distance metric, considering only the rotation matrices in the relevant links homogenous
            transformation matrices
        """
        # Compute ground truth and predicted forward kinematics
        true_fk = self.link_fk_batch(self.urdf, cfgs=real_joint_states, links=self.relevant_links)
        pred_fk = self.link_fk_batch(self.urdf, cfgs=pred_joint_states, links=self.relevant_links)

        # Compute 3D position and rotation errors for each of the relevant links.
        cartesian_loss = torch.zeros(len(true_fk), dtype=torch.float32)
        # Cartesian position error
        pos_error = torch.zeros(len(true_fk), dtype=torch.float32)

        for i, link in enumerate(true_fk):
            # Ignore irrelevant links
            if self.relevant_links is not None and link.name not in self.relevant_links:
                continue

            true_cfg = true_fk[link]
            pred_cfg = pred_fk[link]

            if self.loss_type == 'frobenius':
                # Take into account the rotation and translation coordinates of the joint origin
                error = true_cfg - pred_cfg
                batch_errors = torch.norm(error, p='fro', dim=[-1, -2])
                link_pos_error = torch.mean(batch_errors)
            elif self.loss_type == 'rot_only':
                # Take into account the rotation and translation coordinates of the joint origin
                identity = torch.eye(3, dtype=torch.float32).repeat(true_cfg.shape[0], true_cfg.shape[1], 1, 1)
                error = identity - true_cfg[..., :3, :3].matmul(torch.transpose(pred_cfg[..., :3, :3], -2, -1))
                batch_errors = torch.norm(error, p='fro', dim=[-1, -2])
                link_pos_error = torch.mean(batch_errors)
            elif self.loss_type == 'rot_loc':
                raise NotImplementedError("Rot Loc not yet implemented")

            if self.return_pos_error:
                error = true_cfg[..., :3, 3] - pred_cfg[..., :3, 3]
                batch_pos_errors = torch.norm(error, p='fro', dim=[-1, -2])
                pos_error[i] = torch.mean(batch_pos_errors)

            # Add up the error for each joint
            cartesian_loss[i] = link_pos_error
            # print("- %-20s: L2 error: %.3e " % (link.name, link_pos_error))

        n_links = len(self.relevant_links) if self.relevant_links is not None else len(true_fk)
#         avg_cartesian_loss = torch.true_divide(torch.sum(cartesian_loss), n_links)
        avg_cartesian_loss = torch.div(torch.sum(cartesian_loss), n_links)

        # avg_cartesian_loss.backward()
        if self.return_pos_error:
#             avg_pos_error = torch.true_divide(torch.sum(pos_error), n_links)
            avg_pos_error = torch.div(torch.sum(pos_error), n_links)
            return avg_cartesian_loss, avg_pos_error
        return avg_cartesian_loss

    @staticmethod
    def rotation_matrices(angles, axis):
        """Compute rotation matrices from angle/axis representations, using the Rodrigues rotation formula

        Parameters
        ----------
        angles : (n,) torch.Tensor
            The angles values produced by the network output
        axis : (3,) float
            The axis of the actuated joint. A constant in the computational graph

        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        assert isinstance(angles,
                          torch.Tensor), "Angle values must be instance of torch.Tensor to ensure backpropagation"
        
        batch_size = angles.shape[0]
        n_cfgs = angles.shape[1]     # Number of joint configurations/angles
        axis = torch.from_numpy(axis / np.linalg.norm(axis)).type(torch.float32)
        axis = axis.cuda()
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4).repeat(batch_size, n_cfgs, 1, 1).cuda()
        M[:, :, 0, 0] = cosa
        M[:, :, 1, 1] = cosa
        M[:, :, 2, 2] = cosa
#         import pdb; pdb.set_trace()
        M[:, :, :3, :3] += torch.ger(axis, axis).repeat(batch_size, n_cfgs, 1, 1) * torch.unsqueeze(torch.unsqueeze(1.0 - cosa, -1), -1)

        M[:, :, :3, :3] += torch.FloatTensor(
            [[0.0, -axis[2], axis[1]],
             [axis[2], 0.0, -axis[0]],
             [-axis[1], axis[0], 0.0]]).repeat(batch_size, n_cfgs, 1, 1).cuda() * torch.unsqueeze(torch.unsqueeze(sina, -1), -1)
        return M

    @staticmethod
    def get_child_poses(joint, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given SET of
        configuration values.

        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint loss_type as follows:

            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.

            If ``cfg`` is ``None``, then this just returns the joint pose.

        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return torch.from_numpy(joint.origin).repeat(n_cfgs, 1, 1)
        elif joint.joint_type == 'fixed':
            return torch.from_numpy(joint.origin).repeat(n_cfgs, 1, 1)
        elif joint.joint_type in ['revolute', 'continuous']:
            if cfg is None:
                cfg = torch.zeros(n_cfgs)

            joint_origin = torch.from_numpy(joint.origin.astype(np.float32)).cuda()
            R = CartesianSpaceLoss.rotation_matrices(cfg, joint.axis)
            return joint_origin.matmul(R)
        elif joint.joint_type == 'prismatic':
            raise NotImplementedError("Joint loss_type [%s] not implemented" % joint.joint_type)
        elif joint.joint_type == 'planar':
            raise NotImplementedError("Joint loss_type [%s] not implemented" % joint.joint_type)
        elif joint.joint_type == 'floating':
            raise NotImplementedError("Joint loss_type [%s] not implemented" % joint.joint_type)
        else:
            raise ValueError('Invalid configuration')

    @staticmethod
    def link_fk_batch(urdf, cfgs=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            joint names rather than the links themselves.

        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base joint's frame, or a single
            nx4x4 matrix if ``joint`` is specified.
        """

        # TODO parse to Torch assume they are torch tensors inside
        joint_cfgs, batch_size, n_cfgs = CartesianSpaceLoss.process_cfgs(urdf, cfgs)

        # Process joint set
        link_set = urdf.links
        if links is not None:  # Remove non requested links from set
            for link in list(link_set):
                a = link.name
                if link.name in links:
                    continue
                else:
                    link_set.remove(link)

        # Compute FK mapping each joint to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in urdf._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, dtype=torch.float32).repeat(batch_size, n_cfgs, 1, 1).cuda()
            poses.requires_grad = True
            poses.retain_grad()
            path = urdf._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = urdf._G.get_edge_data(child, parent)['joint']

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = urdf._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]

                # Compute translation and rotation of links given the configurations
                poses = CartesianSpaceLoss.get_child_poses(joint, cfg_vals, n_cfgs).matmul(poses)

                if parent in fk:
                    poses = fk[parent].matmul(poses)
                    break
            fk[lnk] = poses

        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    @staticmethod
    def process_cfgs(urdf, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.

        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {joint: [] for joint in urdf.actuated_joints}
        urdf_joint_names = [joint.name for joint in urdf.actuated_joints]

        # Number of configurations per joint to consider
        batch_size = None
        n_cfgs = None
        if isinstance(cfgs, dict):
            for joint_name in cfgs:
                assert len(cfgs[joint_name].shape) == 2, "Joint configurations must have batch dimension " \
                                                    "(BATCH, JOINT_CONFIGURATIONS)"
                # Find URDF Joint instance corresponding to this cfg
                assert joint_name in urdf_joint_names, "Joint %s not present in URDF" % joint_name
                joint = list(joint_cfg.keys())[urdf_joint_names.index(joint_name)]
                joint_cfg[joint] = cfgs[joint_name]
                if n_cfgs is None:
                    n_cfgs = cfgs[joint_name].shape[1]
                if batch_size is None:
                    batch_size = cfgs[joint_name].shape[0]
        else:
            raise ValueError('Incorrectly formatted joint configurations, required dict mapping joint names and '
                             'joint configurations as tensors')

        for j in joint_cfg:
            if joint_cfg[j].shape[1] == 0:
                joint_cfg[j] = None
            elif joint_cfg[j].shape[1] != n_cfgs:
                raise ValueError('Inconsistent number of configurations for joints')
            elif joint_cfg[j].shape[0] != batch_size:
                raise ValueError('Inconsistent batch_size for joints')

        return joint_cfg, batch_size, n_cfgs

