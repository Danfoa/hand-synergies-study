import rospy
import numpy
import scipy
import pandas
import math
import time 
from enum import Enum

import matplotlib
# matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt

from utils.visualization import *
from utils.data_loader_kine_adl import *

from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock

scale_factor = 1
reflect = 1

passive_joints = ['index_distal_joint', 'middle_distal_joint', 'ring_distal_joint', 'little_distal_joint']

class RightHandJointNames(Enum):
    cmc1_f = 'rh_' + 'thumb_proximal_joint' # Flexion+ / Extension -
    cmc1_a = 'rh_' + 'thumb_abduction_joint' # Abduction+ / Adduction-
    mpc1_f = 'rh_' + 'thumb_middle_joint'
    ip1_f = 'rh_' + 'thumb_distal_joint'
    mcp2_f = 'rh_' + 'index_proximal_joint'
    mcp23_a = 'rh_' + 'index_abduction_joint'
    pip2_f = 'rh_' + 'index_middle_joint'
    mcp3_f = 'rh_' + 'middle_proximal_joint'
    pip3_f = 'rh_' + 'middle_middle_joint'
    mcp4_f = 'rh_' + 'ring_proximal_joint'
    mcp34_a = 'rh_' + 'middle_abduction_joint'
    pip4_f = 'rh_' + 'ring_middle_joint'
    # palm_arch = 'R_PalmArch'
    mcp5_f = 'rh_' + 'little_proximal_joint'
    mcp45_a = 'rh_' + 'ring_abduction_joint'
    pip5_f = 'rh_' + 'little_middle_joint'
    # wr_f = 'R_WR_F'   # Data missing in exp 1
    # wr_a = 'R_WR_A'   # Data missing in exp 1

class LeftHandJointNames(Enum):
    cmc1_f = 'lh_' + 'thumb_proximal_joint'  # Flexion+ / Extension -
    cmc1_a = 'lh_' + 'thumb_abduction_joint'  # Abduction+ / Adduction-
    mpc1_f = 'lh_' + 'thumb_middle_joint'
    ip1_f = 'lh_' + 'thumb_distal_joint'
    mcp2_f = 'lh_' + 'index_proximal_joint'
    mcp23_a = 'lh_' + 'index_abduction_joint'
    pip2_f = 'lh_' + 'index_middle_joint'
    mcp3_f = 'lh_' + 'middle_proximal_joint'
    pip3_f = 'lh_' + 'middle_middle_joint'
    mcp4_f = 'lh_' + 'ring_proximal_joint'
    mcp34_a = 'lh_' + 'middle_abduction_joint'
    pip4_f = 'lh_' + 'ring_middle_joint'
    # palm_arch = 'R_PalmArch'
    mcp5_f = 'lh_' + 'little_proximal_joint'
    mcp45_a = 'lh_' + 'ring_abduction_joint'
    pip5_f = 'lh_' + 'little_middle_joint'
    # wr_f = 'R_WR_F'   # Data missing in exp 1
    # wr_a = 'R_WR_A'   # Data missing in exp 1


def create_joint_state_msg(recordings, hand_keys, right_hand=True):
    assert isinstance(hand_keys, RightHand.__class__) or isinstance(hand_keys, LeftHand.__class__)

    js = JointState()
    # Set time to recording simulated time
    joint_names = RightHandJointNames if right_hand else LeftHandJointNames
    prefix = 'rh_' if right_hand else 'lh_'
    abduction_joints = ['index_abduction_joint', 'middle_abduction_joint', 
                        'ring_abduction_joint','little_abduction_joint']

    # Set normal joint position values directly 
    for joint in joint_names:
        # print(joint.value)
        # print([prefix + s for s in abduction_joints])
        if not joint.value in [prefix + s for s in abduction_joints]:
            js.name.append(joint.value)  # Add ROS joint name
            js.velocity.append(0)
            js.effort.append(0)
            js.position.append(recordings[hand_keys[joint.name].value] * math.pi/180 )
        # else:
        #     print(joint.value)

    reflect = 1 #if right_hand else -1

    js.name.append(prefix + 'index_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append(recordings[hand_keys.mcp23_a.value] * math.pi/180*-1)

    js.name.append(prefix + 'middle_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append(0)

    js.name.append(prefix + 'ring_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append((recordings[hand_keys.mcp34_a.value] * math.pi/180) * reflect)
        
    js.name.append(prefix + 'little_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    js.position.append((recordings[hand_keys.mcp45_a.value] + recordings[hand_keys.mcp34_a.value]) * math.pi/180 * reflect)

    # Use estimated DIP joint angles following the regression models obtained in"
    # "Across-subject calibration of an instrumented glove to measure hand movement for clinical purposes" Verónica Gracia-Ibáñez et.al.
    js.name.append(prefix + 'index_distal_joint')
    estimated_angle = 0.87 * recordings[hand_keys.pip2_f.value] - 25.27
    js.position.append(estimated_angle * math.pi/180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append(prefix + 'middle_distal_joint')
    estimated_angle = 0.79 * recordings[hand_keys.pip3_f.value] - 18.33
    js.position.append(estimated_angle * math.pi/180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append(prefix + 'ring_distal_joint')
    estimated_angle = 0.73 * recordings[hand_keys.pip4_f.value] - 20.54
    js.position.append(estimated_angle * math.pi/180)
    js.velocity.append(0)
    js.effort.append(0)

    js.name.append(prefix + 'little_distal_joint')
    estimated_angle = 0.84 * recordings[hand_keys.pip5_f.value] - 12.42
    js.position.append(estimated_angle * math.pi/180)
    js.velocity.append(0)
    js.effort.append(0)

    return js


if __name__ == '__main__':
    rospy.init_node('manipulation_visualizer')

    # Declare publishers 
    time_pub = rospy.Publisher('/clock', Clock, queue_size=100)
    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=100)

    df = load_subject_data(DATABASE_PATH, subject_id=13, experiment_number=2,
                           task_id=None, records_id=None)
    # plot_task_data(df)
    # print(df.shape)
    # Get all right hand joint positions over time 
    right_hand_data = df[[ExperimentFields.time.value] + [e.value for e in RightHand]]
    left_hand_data = df[[ExperimentFields.time.value] + [e.value for e in LeftHand]]

    # keep iteration count 
    iter = 0
    time_offset = 0.

    plot_data = False
    if plot_data:
        # Configure ploting
        plt.ion()
        fig, ax1, ax2 = get_configured_plot(df)
        fig.set_size_inches(10, 7)
        ax1.set_xlim(left=df.at[0, ExperimentFields.time.value],
                     right=df.loc[df.shape[0] - 1, ExperimentFields.time.value] + 3,
                     auto=False)
        ax1.set_ylim(bottom=-80, top=90, auto=False)
        ax2.set_ylim(bottom=-80, top=90, auto=False)

        right_lines = ax1.plot(right_hand_data.loc[:1, ExperimentFields.time.value].values,
                               right_hand_data.loc[:1, [e.value for e in RightHand]].values)
        left_lines = ax2.plot(left_hand_data.loc[:1, ExperimentFields.time.value].values,
                              left_hand_data.loc[:1, [e.value for e in LeftHand]].values)
        # print(len(right_lines))

        ax1.legend([e.value for e in RightHand])
        ax2.legend([e.value for e in LeftHand])

        background_ax1 = fig.canvas.copy_from_bbox(ax1.bbox)
        background_ax2 = fig.canvas.copy_from_bbox(ax2.bbox)
        fig.canvas.draw()
        fig.show()

    while not rospy.is_shutdown():
        # get sim time 
        # print("%d  Time: %.3f" % (iter, df.at[iter, ExperimentFields.time.value]))
        sim_time = rospy.Time.from_sec(df.at[iter, ExperimentFields.time.value] + time_offset) 

        # Published simulated time stamp 
        # time_msg = Clock()
        # time_msg.clock = sim_time
        # time_pub.publish(time_msg)

        p_time = rospy.get_rostime()

        # Create joint state msg for RIGHT HAND
        js_right = create_joint_state_msg(df.iloc[iter, :], RightHand, right_hand=True)
        js_right.header.stamp = p_time
        js_right.header.seq = iter
        js_pub.publish(js_right)

        # Create joint state msg for LEFT HAND
        js_left = create_joint_state_msg(df.iloc[iter, :], LeftHand, right_hand=False)
        js_left.header.stamp = p_time
        js_left.header.seq = iter
        js_pub.publish(js_left)

        if iter == df.shape[0]-1:
            rospy.sleep(5)
            iter = 0

        if plot_data:
            for line, key in zip(right_lines, RightHand):
                start_t = 0
                if iter > 50:
                    start_t = iter - 50
                line.set_xdata(df.loc[start_t:iter + 1, ExperimentFields.time.value])
                line.set_ydata(df.loc[start_t:iter + 1, key.value])
                ax1.draw_artist(line)
            for line, key in zip(left_lines, LeftHand):
                start_t = 0
                if iter > 50:
                    start_t = iter - 50
                line.set_xdata(df.loc[start_t:iter + 1, ExperimentFields.time.value])
                line.set_ydata(df.loc[start_t:iter + 1, key.value])
                ax2.draw_artist(line)
            # restore background
            # fig.canvas.restore_region(background_ax1)
            # fig.canvas.restore_region(background_ax2)
            # fill in the axes rectangle
            # fig.canvas.blit(ax1.bbox)
            fig.canvas.flush_events()
        # plt.pause(sleep_time)
        # print("%.4e - %.4e = %4e" % (df.at[iter+1, ExperimentFields.time.value],  df.at[iter, ExperimentFields.time.value],
                                    # df.at[iter+1, ExperimentFields.time.value] - df.at[iter, ExperimentFields.time.value]))

        records_dt = df.at[iter+1, ExperimentFields.time.value] - df.at[iter, ExperimentFields.time.value]
        enlapsed_dt = rospy.get_time() - p_time.secs
        print("%.4f - %.4e = %.4f %d" % (records_dt, enlapsed_dt, records_dt-enlapsed_dt, records_dt-enlapsed_dt > 0))

        if records_dt - enlapsed_dt > 0:
            rospy.sleep((records_dt - enlapsed_dt)*2)
        iter += 1
    rospy.signal_shutdown('done')
