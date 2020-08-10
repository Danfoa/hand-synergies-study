import math
from enum import Enum

from utils.data_loader_kine_adl import RightHand

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
    estimated_angle = 0.84 * recordings[hand_keys.pip4_f.value] - 12.42
    js.position.append(estimated_angle * math.pi / 180)
    js.velocity.append(0)
    js.effort.append(0)

    return js
