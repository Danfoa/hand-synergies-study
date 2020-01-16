# Python includes
import numpy
import random
import cmd 
from math import pi
import time

from sklearn.linear_model import LinearRegression

# ROS includes
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Polygon
from tf import transformations # rotation_matrix(), concatenate_matrices()

from sensor_msgs.msg import JointState
from vmg30_driver.msg import RawSensorOutput 

import rviz_tools_py as rviz_tools


joint_names = ["thumb_abduction_joint", "thumb_proximal_joint", "thumb_middle_joint", "thumb_distal_joint",
               "index_abduction_joint", "index_proximal_joint", "index_middle_joint", "index_distal_joint",
               "middle_abduction_joint", "middle_proximal_joint", "middle_middle_joint", "middle_distal_joint",
               "ring_abduction_joint", "ring_proximal_joint", "ring_middle_joint", "ring_distal_joint",
               "little_abduction_joint", "little_proximal_joint", "little_middle_joint", "little_distal_joint"]

joint_sensors = {
                "thumb_abduction_joint" : [""],
                "thumb_proximal_joint" : [""], 
                "thumb_middle_joint" : [""], 
                "thumb_distal_joint" : [""],
                "index_abduction_joint" : [""], 
                "index_proximal_joint" : ["indexPh1"], 
                "index_middle_joint" : ["indexPh2"], 
                "index_distal_joint" : [""],
                "middle_abduction_joint" : [""], 
                "middle_proximal_joint" : ["middlePh1"], 
                "middle_middle_joint" : ["middlePh2"], 
                "middle_distal_joint" : [""],
                "ring_abduction_joint" : [""], 
                "ring_proximal_joint" : ["ringPh1"], 
                "ring_middle_joint" : ["ringPh2"], 
                "ring_distal_joint" : [""],
                "little_abduction_joint" : [""], 
                "little_proximal_joint" : ["littlePh1"], 
                "little_middle_joint" : ["littlePh2"], 
                "little_distal_joint" : [""]
                }

sensor_ids = None
sensor_values = numpy.array(RawSensorOutput().sensor_values)

# Define exit handler
def cleanup_node():
    print "Shutting down node"
    markers.deleteAllMarkers()


def show_instruction(msg):
    global markers
    text_pose = Pose(Point(0, 0, 0.25), Quaternion(0, 0, 0, 1))
    scale = Vector3(0.02, 0.02, 0.02)
    markers.deleteAllMarkers()
    markers.publishText(text_pose, msg, color='white', scale=scale, lifetime=None)

def update_glove_data(incoming_data):
    global sensor_ids, sensor_values
    sensor_ids = numpy.array(incoming_data.sensor_ids)
    sensor_values = numpy.array(incoming_data.sensor_values)



def get_avg_sensor_values():
    global sensor_values

    rate = rospy.Rate(30)
    start_time = time.time()
    sensor_readings = numpy.array(sensor_values)
    count = 1
    while(time.time() - start_time < 1):
        sensor_readings += numpy.array(sensor_values)
        count +=1
        rate.sleep()
    return sensor_readings / count 


def update_visualization_pose(pub, positions):
    global joint_names

    js = JointState()
    js.name = joint_names
    js.position = positions
    js.effort = [0] * len(joint_names)
    js.velocity = [0] * len(joint_names)
    js_pub.publish(js)

markers = rviz_tools.RvizMarkers('/world', 'visualization_marker')

if __name__ == '__main__':
    rospy.init_node('glove_calibration')
    # Clean node at shutdown 
    rospy.on_shutdown(cleanup_node)
    rospy.Subscriber("/sensor_data", RawSensorOutput, update_glove_data)
    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=100)
    update_visualization_pose(js_pub, positions=list(numpy.zeros((len(joint_names),))))

    while not rospy.is_shutdown() and sensor_ids is None:
        print("Waiting for sensor msgs...", sensor_ids)
        rospy.sleep(1)

    # Publish some text using a ROS Pose Msg
    
    show_instruction('Press enter to start the calibration process')
    _ = raw_input("Please ENTER to start the calibration process continue")
    show_instruction("Perform the requested finger movements.\n" \
                    "press enter once you have the fingers on the correct position \n" \
                    "and hold for two seconds.")
    _ = raw_input("Please ENTER to continue")

    # MCP [0, 45 ,75] PIP 0 degrees -----------------------------------------------------------
    mcp_names = ["index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint", "little_proximal_joint"]
    mcp_models = []

    for mcp_joint in mcp_names:
        y_train = numpy.expand_dims(numpy.array([0, 45, 75], dtype=float), axis=1)
        # Allocate matrix with shape (num of target angles, num of sensors)
        x_train = None
        for mcp_angle in y_train:
            
            show_instruction('Flexion of \"%s\" at %d degrees\npress ENTER when ready' % (mcp_joint, mcp_angle))
            
            # Set desired pose on visualization.
            joint_index = joint_names.index(mcp_joint) 
            target_angles = numpy.zeros((len(joint_names),))
            target_angles[joint_index] = mcp_angle * pi / 180 
            update_visualization_pose(js_pub, positions=list(target_angles))
            _ = raw_input("Please ENTER to continue")
            show_instruction("Hold position...")

            # Obtain average value of sensor data for one second. 
            sensor_values = get_avg_sensor_values()

            # Filter sensor values to consider only joint relevant sensors. 
            relevant_sensors = joint_sensors[mcp_joint]
            mask = numpy.zeros((len(sensor_ids),))
            print([numpy.where(sensor_ids == s)[0][0] for s in relevant_sensors])
            mask[[numpy.where(sensor_ids == s)[0][0] for s in relevant_sensors]] = 1
            # mask[[numpy.wheresensor_ids== for s in relevant_sensors]] = 1
            sensor_values = numpy.multiply(sensor_values, mask)
            # sensor_values = sensor_values[[numpy.where(sensor_ids == s)[0] for s in relevant_sensors]]
            x_train = sensor_values if x_train is None else numpy.vstack((x_train, sensor_values))

        print("Data collected from [%s]" % (mcp_joint))
        print(x_train.shape ,  y_train.shape)


        model = LinearRegression()
        print(x_train)
        print(y_train)
        model.fit(X=x_train, y=y_train)
        mcp_models.append(model)
        print("Model fitted")
        # print(model.coef_, model.intercept_)
  
    mcp_matrix = numpy.vstack(tuple([m.coef_ for m in mcp_models]))
    mcp_bias = numpy.vstack(tuple([m.intercept_ for m in mcp_models]))

     # MCP [0, 45 ,75] PIP 0 degrees -----------------------------------------------------------
    pip_names = ["index_middle_joint", "middle_middle_joint", "ring_middle_joint", "little_middle_joint"]
    pip_models = []

    for pip_joint in pip_names:
        y_train = numpy.expand_dims(numpy.array([0, 45, 75], dtype=float), axis=1)
        # Allocate matrix with shape (num of target angles, num of sensors)
        x_train = None
        for pip_angle in y_train:
            
            show_instruction('Flexion of \"%s\" at %d degrees\npress ENTER when ready' % (pip_joint, pip_angle))
            
            # Set desired pose on visualization.
            joint_index = joint_names.index(pip_joint) 
            target_angles = numpy.zeros((len(joint_names),))
            target_angles[joint_index] = pip_angle * pi / 180 
            update_visualization_pose(js_pub, positions=list(target_angles))
            _ = raw_input("Please ENTER to continue")
            show_instruction("Hold position...")

            # Obtain average value of sensor data for one second. 
            sensor_values = get_avg_sensor_values()

            # Filter sensor values to consider only joint relevant sensors. 
            relevant_sensors = joint_sensors[pip_joint]
            mask = numpy.zeros((len(sensor_ids),))
            print([numpy.where(sensor_ids == s)[0][0] for s in relevant_sensors])
            mask[[numpy.where(sensor_ids == s)[0][0] for s in relevant_sensors]] = 1
            # mask[[numpy.wheresensor_ids== for s in relevant_sensors]] = 1
            sensor_values = numpy.multiply(sensor_values, mask)
            # sensor_values = sensor_values[[numpy.where(sensor_ids == s)[0] for s in relevant_sensors]]
            x_train = sensor_values if x_train is None else numpy.vstack((x_train, sensor_values))

        print("Data collected from [%s]" % (pip_joint))
        print(x_train.shape ,  y_train.shape)


        model = LinearRegression()
        print(x_train)
        print(y_train)
        model.fit(X=x_train, y=y_train)
        pip_models.append(model)
        print("Model fitted")
        # print(model.coef_, model.intercept_)

    pip_matrix = numpy.vstack(tuple([m.coef_ for m in pip_models]))
    pip_bias = numpy.vstack(tuple([m.intercept_ for m in pip_models]))          
    

    while not rospy.is_shutdown():
        # print("IndexMCP %.3f" % model.predict(numpy.expand_dims(sensor_values,axis=0)))
        
        mcp_angles = numpy.matmul(mcp_matrix, numpy.expand_dims(sensor_values,axis=1)) + mcp_bias
        pip_angles = numpy.matmul(pip_matrix, numpy.expand_dims(sensor_values,axis=1)) + pip_bias

        angles = numpy.zeros((len(joint_names),))
        angles[5] = mcp_angles[0]
        angles[9] = mcp_angles[1]
        angles[13] = mcp_angles[2]
        angles[17] = mcp_angles[3]

        angles[6] = pip_angles[0]
        angles[10] = pip_angles[1]
        angles[14] = pip_angles[2]
        angles[18] = pip_angles[3]

        angles[7] = pip_angles[0]  * (2/3)
        angles[11] = pip_angles[1] * (2/3)
        angles[15] = pip_angles[2] * (2/3)
        angles[19] = pip_angles[3] * (2/3)

        js = JointState()
        js.name = joint_names
        js.position = angles*pi/180
        js_pub.publish(js)


