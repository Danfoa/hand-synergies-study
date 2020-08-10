import math
from enum import Enum
import torch

from .data_loader_kine_adl import RightHand

scale_factor = 1
reflect = 1

passive_joints = ['index_distal_joint', 'middle_distal_joint', 'ring_distal_joint', 'little_distal_joint']


class RightHandJointNames(Enum):
    cmc1_f = 'thumb_proximal_joint'  # Flexion+ / Extension -
    cmc1_a = 'thumb_abduction_joint'  # Abduction+ / Adduction-
    mpc1_f = 'thumb_middle_joint'
    ip1_f = 'thumb_distal_joint'
    mcp2_f = 'index_proximal_joint'
    mcp23_a = 'index_abduction_joint'
    pip2_f = 'index_middle_joint'
    mcp3_f = 'middle_proximal_joint'
    pip3_f = 'middle_middle_joint'
    mcp4_f = 'ring_proximal_joint'
    mcp34_a = 'middle_abduction_joint'
    pip4_f = 'ring_middle_joint'
    # palm_arch = 'R_PalmArch'    #   URDF model does not has this angle joint
    mcp5_f = 'little_proximal_joint'
    mcp45_a = 'ring_abduction_joint'
    pip5_f = 'little_middle_joint'
    wr_f = 'palm_joint'   # Data missing in exp 1
    wr_a = 'palm_joint_abduction'   # Data missing in exp 1


# Naive implementation of ROS JointState to not loose the ROS structure
class JointState:

    def __init__(self):
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


def create_joint_state_msg(recordings, hand_keys):
    assert isinstance(hand_keys, RightHand.__class__)

    js = JointState()
    # Set time to recording simulated time
    joint_names = RightHandJointNames
    abduction_joints = ['index_abduction_joint', 'middle_abduction_joint',
                        'ring_abduction_joint', 'little_abduction_joint']

    # Set normal joint position values directly
    for joint in joint_names:
        # print(joint.value)
        # print([s for s in abduction_joints])
        if not joint.value in abduction_joints:
            js.name.append(joint.value)  # Add ROS joint name
            js.velocity.append(0)
            js.effort.append(0)
            js.position.append(recordings[hand_keys[joint.name].value] * math.pi / 180)

    reflect = 1
    # ABDUCTION JOINTS
    js.name.append('index_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append(recordings[hand_keys.mcp23_a.value] * math.pi / 180 * -1)

    js.name.append('middle_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append(0)

    js.name.append('ring_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append((recordings[hand_keys.mcp34_a.value] * math.pi / 180) * reflect)

    js.name.append('little_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append(
        (recordings[hand_keys.mcp45_a.value] + recordings[hand_keys.mcp34_a.value]) * math.pi / 180 * reflect)

    # DIP `passive` JOINTS
    js.name.append('index_distal_joint')
    estimated_angle = 0.87 * recordings[hand_keys.pip2_f.value] - 25.27
    js.position.append(estimated_angle * math.pi / 180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append('middle_distal_joint')
    estimated_angle = 0.79 * recordings[hand_keys.pip3_f.value] - 18.33
    js.position.append(estimated_angle * math.pi / 180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append('ring_distal_joint')
    estimated_angle = 0.73 * recordings[hand_keys.pip4_f.value] - 20.54
    js.position.append(estimated_angle * math.pi / 180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append('little_distal_joint')
    estimated_angle = 0.84 * recordings[hand_keys.pip5_f.value] - 12.42
    js.position.append(estimated_angle * math.pi / 180)
    js.velocity.append(0)
    js.effort.append(0)

    return js


def joint_data_to_urdf_joint_state(data, data_keys, hand_keys):
    """
    Utility function to convert the ouput of the NN predicted joint angle values in deg to the URDF joint state space
    equivalent for the hand model.
    :param data: Matrix of joint angle values following the ADL dataset convention in deg, of shape
    (Batch, Window_size, Joints)
    :type data: torch.Tensor
    :param data_keys: An ordered list with the names of the joints (following the ADL dataset convention) present in
    the last dimension of the `data`
    :param hand_keys: An instance of RightHand or LeftHand enum types
    :return: A dictionary contaning a mapping from URDF joint names to the torch.Tensor values provided in Data,
    appropriately formated and scaled
    """
    assert isinstance(data, torch.Tensor), "data should be a Tensor"

    joint_names = RightHandJointNames
    abduction_joints = ['index_abduction_joint', 'middle_abduction_joint',
                        'ring_abduction_joint', 'little_abduction_joint']

    urdf_cfgs = {}
    rad_data = data * (math.pi/180)

    # Set normal joint position values directly
    for joint in joint_names:
        if not joint.value in abduction_joints:
            adl_dataset_joint_name = hand_keys[joint.name].value
            joint_data_idx = data_keys.index(adl_dataset_joint_name)
            urdf_cfgs[joint.value] = rad_data[..., joint_data_idx]

    reflect = 1
    # ABDUCTION JOINTS
    adl_dataset_joint_name = hand_keys.mcp23_a.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['index_abduction_joint'] = rad_data[..., joint_data_idx] * -reflect

    # Assume fixed middle abduction joint
    urdf_cfgs['middle_abduction_joint'] = rad_data[..., 0] * 0

    adl_dataset_joint_name = hand_keys.mcp34_a.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['ring_abduction_joint'] = rad_data[..., joint_data_idx] * reflect

    adl_dataset_joint_name = hand_keys.mcp45_a.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    joint_values = rad_data[..., joint_data_idx]
    adl_dataset_joint_name = hand_keys.mcp34_a.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    joint_values += rad_data[..., joint_data_idx]
    urdf_cfgs['little_abduction_joint'] = joint_values * reflect

    # DIP `passive` JOINTS
    adl_dataset_joint_name = hand_keys.pip2_f.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['index_distal_joint'] = (0.87 * data[..., joint_data_idx] - 25.27) * (math.pi/180)

    adl_dataset_joint_name = hand_keys.pip3_f.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['middle_distal_joint'] = (0.79 * data[..., joint_data_idx] - 18.33) * (math.pi/180)

    adl_dataset_joint_name = hand_keys.pip4_f.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['ring_distal_joint'] = (0.73 * data[..., joint_data_idx] - 20.54) * (math.pi/180)

    adl_dataset_joint_name = hand_keys.pip5_f.value
    joint_data_idx = data_keys.index(adl_dataset_joint_name)
    urdf_cfgs['little_distal_joint'] = (0.84 * data[..., joint_data_idx] - 12.42) * (math.pi/180)

    return urdf_cfgs