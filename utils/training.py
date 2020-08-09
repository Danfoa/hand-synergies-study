from collections import OrderedDict

import torch
import numpy as np
from urdfpy import URDF
import warnings


class CartesianSpaceLoss(torch.nn.Module):

    def __init__(self, urdf, type='frobenius', relevant_links=None, return_pos_error=False):
        super(CartesianSpaceLoss, self).__init__()

        if not isinstance(urdf, URDF):
            raise ValueError("Invalid urdf argument")

        if relevant_links is not None:
            link_names = [link.name for link in urdf.links]
            for relevant_link in relevant_links:
                if relevant_link not in link_names:
                    raise ValueError("Relevant link %s not present in provided `urdf`" % relevant_link)
            print("Loss will consider the following links %s " % relevant_links)

        if type not in ['rot_only', 'frobenius', 'rot_loc']:
            raise ValueError("Invalid loss `type` use: 'rot_only', 'frobenius' or 'rot_loc'")

        self.urdf = urdf
        self.loss_type = type
        self.relevant_links = relevant_links
        self.return_pos_error = return_pos_error

    def forward(self, pred_joint_states, real_joint_states):
        """
        Forward pass of the loss function using the predicted and true joint states to estimate a distance metric of the
        robot/system links through the use of forward kinematics.
        :param pred_joint_states: Predicted joint angle values of all actuated joints
        :type pred_joint_states: dict
        :param real_joint_states: Ground truth joint angle values of all actuated joints
        :type real_joint_states: dict
        :return: A distance scalar metric in cartesian space. If the loss `type` is `frobenius` the returned distance
            scalar is the average Frobenius norm across all `relevant_links` homogenous transformation matrices. If type
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
                batch_errors = torch.norm(error, p='fro', dim=[1, 2])
                link_pos_error = torch.mean(batch_errors)
            elif self.loss_type == 'rot_only':
                # Take into account the rotation and translation coordinates of the joint origin
                identity = torch.eye(3, dtype=torch.float32).repeat(true_cfg.shape[0], 1, 1)
                error = identity - true_cfg[:, :3, :3].matmul(torch.transpose(pred_cfg[:, :3, :3], 1, 2))
                batch_errors = torch.norm(error, p='fro', dim=[1, 2])
                link_pos_error = torch.mean(batch_errors)
            elif self.loss_type == 'rot_loc':
                raise NotImplementedError("Rot Loc not yet implemented")

            if self.return_pos_error:
                error = true_cfg[:, :3, 3] - pred_cfg[:, :3, 3]
                batch_pos_errors = torch.norm(error, p='fro', dim=[1, 2])
                pos_error[i] = torch.mean(batch_pos_errors)

            # Add up the error for each joint
            cartesian_loss[i] = link_pos_error
            print("- %-20s: L2 error: %.3e " % (link.name, link_pos_error))

        n_links = len(self.relevant_links) if self.relevant_links is not None else len(true_fk)
        avg_cartesian_loss = torch.true_divide(torch.sum(cartesian_loss), n_links)
        # avg_cartesian_loss.backward()
        if self.return_pos_error:
            avg_pos_error = torch.true_divide(torch.sum(pos_error), n_links)
            return avg_cartesian_loss, avg_pos_error
        return avg_cartesian_loss

    @staticmethod
    def rotation_matrices(angles, axis):
        """Compute rotation matrices from angle/axis representations.

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
        # assert len(angles.shape) == 3, "Angle values shape bust be (BATCH, WINDOW_SIZE, JOINTS)"

        # batch_size = angles.shape[0]
        axis = torch.from_numpy(axis / np.linalg.norm(axis))
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4).repeat(len(angles), 1, 1)
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += torch.ger(axis, axis).repeat(len(angles), 1, 1) * torch.unsqueeze(
            torch.unsqueeze(1.0 - cosa, -1), -1)

        M[:, :3, :3] += torch.FloatTensor(
            [[0.0, -axis[2], axis[1]],
             [axis[2], 0.0, -axis[0]],
             [-axis[1], axis[0], 0.0]]).repeat(len(angles), 1, 1) * torch.unsqueeze(torch.unsqueeze(sina, -1), -1)
        return M

    @staticmethod
    def get_child_poses(joint, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given SET of
        configuration values.

        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:

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

            joint_origin = torch.from_numpy(joint.origin.astype(np.float32))
            R = CartesianSpaceLoss.rotation_matrices(cfg, joint.axis)
            return joint_origin.matmul(R)
        elif joint.joint_type == 'prismatic':
            raise NotImplementedError("Joint type [%s] not implemented" % joint.joint_type)
        elif joint.joint_type == 'planar':
            raise NotImplementedError("Joint type [%s] not implemented" % joint.joint_type)
        elif joint.joint_type == 'floating':
            raise NotImplementedError("Joint type [%s] not implemented" % joint.joint_type)
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
        joint_cfgs, n_cfgs = urdf._process_cfgs(cfgs)

        # Process joint set
        link_set = urdf.links
        if links is not None:  # Remove non requested links from set
            for link in link_set:
                if link.name in links:
                    continue
                else:
                    link_set.remove(link)

        # Compute FK mapping each joint to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in urdf._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, dtype=torch.float32).repeat(n_cfgs, 1, 1)
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
        joint_cfg = {j: [] for j in urdf.actuated_joints}

        # Number of configurations per joint to consider
        n_cfgs = None
        if isinstance(cfgs, dict):
            for joint in cfgs:
                joint_cfg[joint] = cfgs[joint]
                if n_cfgs is None:
                    n_cfgs = len(cfgs[joint])
        elif isinstance(cfgs, (list, tuple, np.ndarray)):
            n_cfgs = len(cfgs)
            if isinstance(cfgs[0], dict):
                for cfg in cfgs:
                    for joint in cfg:
                        joint_cfg[joint].append(cfg[joint])
            elif cfgs[0] is None:
                pass
            else:
                # Matrix based configuration TODO
                cfgs = np.asanyarray(cfgs, dtype=np.float64)
                for i, j in enumerate(urdf.actuated_joints):
                    joint_cfg[j] = cfgs[:, i]
        else:
            raise ValueError('Incorrectly formatted config array')

        for j in joint_cfg:
            if len(joint_cfg[j]) == 0:
                joint_cfg[j] = None
            elif len(joint_cfg[j]) != n_cfgs:
                raise ValueError('Inconsistent number of configurations for joints')

        return joint_cfg, n_cfgs

