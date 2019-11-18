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
from utils.data_loader import *

from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock

scale_factor = 1
reflect = 1


class RightHandJointNames(Enum):
    cmc1_f = 'thumb_proximal_joint' # Flexion+ / Extension -
    cmc1_a = 'thumb_abduction_joint' # Abduction+ / Adduction-
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
    # palm_arch = 'R_PalmArch'
    mcp5_f = 'little_proximal_joint'
    mcp45_a = 'ring_abduction_joint'
    pip5_f = 'little_middle_joint'
    # wr_f = 'R_WR_F'   # Data missing in exp 1
    # wr_a = 'R_WR_A'   # Data missing in exp 1


def create_joint_state_msg(recordings, hand_keys):
    js = JointState()
    # Set time to recording simulated time



    # Set normal joint position values directly 
    for joint in RightHandJointNames:
        if not joint.value in ['index_abduction_joint', 'middle_abduction_joint', 'ring_abduction_joint', 'little_abduction_joint']:
            js.name.append(joint.value)  # Add ROS joint name
            js.velocity.append(0)
            js.effort.append(0)
            js.position.append(recordings[hand_keys[joint.name].value] * math.pi/180 )

    # 	<origin xyz="${0.02175*reflect*scale_factor} ${-0.002*scale_factor} ${0.099*scale_factor}" rpy="0 0 0"/>
    x = 0.02475 * reflect * scale_factor
    y = -0.0125 * scale_factor
    z = 0.0125 * scale_factor
    gamma_index = math.atan2(z, -x)
    # 	<origin xyz="${-0.02225*reflect*scale_factor} ${0.0010*scale_factor} ${0.090*scale_factor}" rpy="0 0 0"/>
    x = -0.02225 * reflect * scale_factor
    y = 0.0010 * scale_factor
    z = 0.090 * scale_factor
    gamma_ring = math.atan2(z, -x)
    # 	<origin xyz="${-0.04175*reflect*scale_factor} ${-0.003*scale_factor} ${0.079*scale_factor}" rpy="0 0 0"/>
    x = -0.04175 * reflect * scale_factor
    y = -0.003 * scale_factor
    z = 0.079 * scale_factor
    gamma_little = math.atan2(z, -x)

    js.name.append('index_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    # js.position.append(0)
    # print((gamma_index - recordings[hand_keys.mcp23_a.value] * math.pi/180)*180/math.pi)
    js.position.append(recordings[hand_keys.mcp23_a.value] * math.pi/180*-1)

    js.name.append('middle_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    # fix this joint value spatula
    js.position.append(0)


    js.name.append('ring_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    # js.position.append(0)
    js.position.append((recordings[hand_keys.mcp34_a.value] * math.pi/180) * 1)
        
    js.name.append('little_abduction_joint')
    js.velocity.append(0)
    js.effort.append(0)
    # js.position.append(0)
    print()
    js.position.append((recordings[hand_keys.mcp45_a.value] + recordings[hand_keys.mcp34_a.value]) * math.pi/180)
    #
    return js


if __name__ == '__main__':
    rospy.init_node('manipulation_visualizer')

    # Declare publishers 
    time_pub = rospy.Publisher('/clock', Clock, queue_size=1)
    js_pub = rospy.Publisher('/db_joint_states', JointState, queue_size=50)

    df = load_subject_data(DATABASE_PATH, subject_id=1, experiment_number=1,
                           task_id=None, records_id=list(range(101, 134)))
    # plot_task_data(df)
    # print(df.shape)
    # Get all right hand joint positions over time 
    right_hand_grasps = df[[ExperimentFields.time.value] + [e.value for e in RightHand]]

    # keep iteration count 
    iter = 0
    time_offset = 0.

    plot_data = True
    if plot_data:
        # Configure ploting
        plt.ion()
        fig, ax1 = plt.subplots(1, 1, sharex='all')
        fig.set_size_inches(10, 7)
        # fig.set_size_
        ax1.set_title('Right Hand')
        ax1.set_ylabel('Angle position [deg]')
        ax1.grid()

        ax1.set_xlim(left=df.at[0, ExperimentFields.time.value],
                     right=df.loc[df.shape[0] - 1, ExperimentFields.time.value]+3,
                     auto=False)
        ax1.set_ylim(bottom=-80, top=90, auto=False)

        lines = ax1.plot(right_hand_grasps.loc[:1, ExperimentFields.time.value].values,
                            right_hand_grasps.loc[:1, [e.value for e in RightHand]].values)
        print(len(lines))

        ax1.legend([e.value for e in RightHand])
        background = fig.canvas.copy_from_bbox(ax1.bbox)
        fig.canvas.draw()
        fig.show()

    while not rospy.is_shutdown():
        # get sim time 
        print("%d  Time: %.3f" % (iter, df.at[iter, ExperimentFields.time.value]))
        sim_time = rospy.Time.from_sec(df.at[iter, ExperimentFields.time.value] + time_offset) 

        # Published simulated time stamp 
        time_msg = Clock()
        time_msg.clock = sim_time
        time_pub.publish(time_msg)

        # Create joint state msgs 
        js_right = create_joint_state_msg(df.iloc[iter,:], RightHand)
        js_right.header.stamp = time_msg.clock
        js_right.header.seq = iter 

        # js_left = create_joint_state_msg(df.iloc[iter,:], LeftHand)
        js_pub.publish(js_right)

        if iter == df.shape[0]-1:
            # time_offset = sim_time.to_sec()
            iter = 0
            print("..")
        sleep_time = df.at[iter+1, ExperimentFields.time.value] - df.at[iter, ExperimentFields.time.value]
        if plot_data:
            for line, key in zip(lines, RightHand):
                line.set_xdata(df.loc[:iter+1, ExperimentFields.time.value])
                line.set_ydata(df.loc[:iter+1, key.value])
                ax1.draw_artist(line)
            # restore background
            # fig.canvas.restore_region(background)
            # fill in the axes rectangle
            # fig.canvas.blit(ax1.bbox)
            fig.canvas.flush_events()
        # plt.pause(sleep_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        iter += 1
    rospy.signal_shutdown('done')
