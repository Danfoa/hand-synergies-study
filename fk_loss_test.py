import numpy as np
import torch

from utils.data_loader_kine_adl import DATABASE_PATH, RightHand, load_subject_data
from utils.data_visualizer_kine_adl import RightHandJointNames, create_joint_state_msg, joint_data_to_urdf_joint_state

from utils.pyrender_visualization import prediction_animation, fixed_prediction_animation

from urdfpy import URDF
from utils.training import CartesianSpaceLoss

if __name__ == "__main__":

    # Load the data
    df = load_subject_data(database_path=DATABASE_PATH,
                           subject_id=8, records_id=[102], load_anatomic_data=False)

    positions = None
    joint_names = None

    for iter in range(df.shape[0]):
        # Create joint state msg for RIGHT HAND
        js_right = create_joint_state_msg(df.iloc[iter, :], RightHand)
        positions = np.vstack([positions, js_right.position]) if positions is not None else js_right.position
        joint_names = js_right.name

    # torch_data = torch.from_numpy(df[])
    adl_right_hand_joint_names = [j.value for j in RightHand]
    right_hand_data = df[adl_right_hand_joint_names]

    urdf = URDF.load('robots/right_hand_relative.urdf')

    WINDOW_SIZE = 10
    BATCH_SIZE = 16
    offset = 3

    # This will simulate the output of the Neural Network
    torch_data = torch.from_numpy(right_hand_data.iloc[:WINDOW_SIZE+offset].values)
    # torch_data = torch_data.repeat(BATCH_SIZE, 1, 1)
    torch_data.requires_grad = True

    # This will configure the network output into a data structure easy to process for FK and convert the joint values
    # of the dataset into the URDF model convention.
    true_cfg = joint_data_to_urdf_joint_state(data=torch_data[:WINDOW_SIZE], data_keys=adl_right_hand_joint_names, hand_keys=RightHand)
    pred_cfg = joint_data_to_urdf_joint_state(data=torch_data[offset:offset+WINDOW_SIZE], data_keys=adl_right_hand_joint_names, hand_keys=RightHand)

    # This is used to check/debug the gradient of each joint prediction values separately
    for _, angle_tensor in pred_cfg.items():
        angle_tensor.retain_grad()

    # As some of the links in our URDF are unactuated, or do not have an equivalent on the dataset, they have to be
    # removed
    urdf_relevant_links = ['palm_base', 'palm_link', 'thumb_knuckle_link', 'thumb_proximal_link',
                           'thumb_middle_link', 'index_knuckle_link', 'index_proximal_link',
                           'index_middle_link', 'middle_knuckle_link', 'middle_proximal_link',
                           'middle_middle_link', 'ring_knuckle_link', 'ring_proximal_link',
                           'ring_middle_link', 'little_knuckle_link', 'little_proximal_link',
                           'little_middle_link']

    # Instantiation of the loss module
    fk_loss = CartesianSpaceLoss(urdf, type='frobenius', relevant_links=urdf_relevant_links)

    # Test that the gradients are being computed and propagated appropriately
    loss = fk_loss(pred_cfg, true_cfg)
    loss.backward()
    print("\nL2 Loss value: %s\n" % loss)
    for joint, angle_values in pred_cfg.items():
        print("- Grad of joint %-25s:  %s" % (joint, angle_values.grad))


# # Test that the forward kinematics in TORCH results in the same fk of the urdfpy package
# fk_pred = fk_loss.link_fk_batch(urdf, cfgs=pred_cfg)
# fk_urdf_pred = urdf.link_fk_batch(cfgs=pred_np_cfg)
# for joint in fk_urdf_pred:
#     true_pos = fk_urdf_pred[joint]
#     torch_pos = fk_pred[joint].detach().numpy()
#     error = np.abs(torch_pos-true_pos)
#     assert np.max(error) < 1e-6, "Link %s forward kinematics mismatch, max error: %.3e" % (joint.name, np.max(error))
#     print("\n Batch Torch based forward kinematics is working appropriately\n")
