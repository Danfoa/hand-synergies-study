import numpy as np
import torch
import timeit
from utils.data_loader_kine_adl import DATABASE_PATH, RightHand, load_subject_data
from utils.data_visualizer_kine_adl import RightHandJointNames, create_joint_state_msg, joint_data_to_urdf_joint_state

from utils.pyrender_visualization import prediction_animation, fixed_prediction_animation

from urdfpy import URDF
from utils.cartesian_space_loss import CartesianSpaceLoss

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

    # nn_output = torch.from_numpy(df[])
    adl_right_hand_joint_names = [j.value for j in RightHand]
    right_hand_data = df[adl_right_hand_joint_names]


    WINDOW_SIZE = 3
    BATCH_SIZE = 1
    offset = 100  # debug variable to generate the simulated data

    # This will simulate the output of the Neural Network _____________________________________________________________
    nn_output = torch.from_numpy(right_hand_data.iloc[offset:offset+WINDOW_SIZE].values)
    nn_output = nn_output.repeat(BATCH_SIZE, 1, 1)
    nn_output.requires_grad = True

    # This will configure the network output into a data structure easy to process for FK and convert the joint values
    # of the dataset into the URDF model convention.
    pred_cfg = joint_data_to_urdf_joint_state(data=nn_output, data_keys=adl_right_hand_joint_names, hand_keys=RightHand)

    # This is used to check/debug the gradient of each joint prediction values separately
    for _, angle_tensor in pred_cfg.items():
        angle_tensor.retain_grad()

    # This will simulate the ground truth joint configurations/angles ________________________________________________
    target_output = torch.from_numpy(right_hand_data.iloc[:WINDOW_SIZE].values)
    target_output = target_output.repeat(BATCH_SIZE, 1, 1)
    target_output.requires_grad = False
    true_cfg = joint_data_to_urdf_joint_state(data=target_output, data_keys=adl_right_hand_joint_names, hand_keys=RightHand)

    # Instantiation of loss module and configuration ________________________________________________________________
    # As some of the links in our URDF do not have an equivalent on the dataset, they have to be removed
    urdf_relevant_links = ['palm_base', 'palm_link', 'thumb_knuckle_link', 'thumb_proximal_link',
                           'thumb_middle_link', 'index_knuckle_link', 'index_proximal_link',
                           'index_middle_link', 'middle_knuckle_link', 'middle_proximal_link',
                           'middle_middle_link', 'ring_knuckle_link', 'ring_proximal_link',
                           'ring_middle_link', 'little_knuckle_link', 'little_proximal_link',
                           'little_middle_link']

    # An instance of the hand URDF is required
    urdf = URDF.load('robots/right_hand_relative.urdf')

    # Instantiation of the loss module, which have two types of loss functions:
    # 1
    fk_loss = CartesianSpaceLoss(urdf, loss_type='frobenius', relevant_links=urdf_relevant_links)
    # 2
    # fk_loss = CartesianSpaceLoss(urdf, loss_type='rot_only', relevant_links=urdf_relevant_links)

    start = timeit.default_timer()
    # Loss forward pass example:
    loss = fk_loss(pred_cfg, true_cfg)
    stop = timeit.default_timer()
    print('\nLoss forward enlapsed time (CPU) for batch size %d and window size %s: %.5fs' % (BATCH_SIZE, WINDOW_SIZE, stop - start))

    # Test that the gradients are being computed and propagated appropriately _____________________________________
    loss.backward()
    print("\nL2 (%s) Loss value: %s" % (fk_loss.loss_type, loss))

    print("\n Grads for the nn_output have shape: %s \n" % list(nn_output.grad.shape))

    # Check that the gradients of the urdf joint state predictions
    for joint, angle_values in pred_cfg.items():
        print("- Grad of joint %-25s:  %s" % (
        joint, angle_values.grad[0] if angle_values.grad is not None else angle_values.grad))

    # Sanity CHECK:
    # Check that the batch torch based forward kinematics is operating correctly __________________________________
    pred_np_cfg = {}
    for joint_name in pred_cfg:
        pred_np_cfg[joint_name] = pred_cfg[joint_name][0].detach().numpy()

    # Test that the forward kinematics in TORCH results in the same fk of the urdfpy package
    fk_pred = fk_loss.link_fk_batch(urdf, cfgs=pred_cfg)
    fk_urdf_pred = urdf.link_fk_batch(cfgs=pred_np_cfg)
    for joint in fk_urdf_pred:
        true_pos = fk_urdf_pred[joint]
        torch_pos = fk_pred[joint][0].detach().numpy()
        error = np.abs(torch_pos-true_pos)
        assert np.max(error) < 1e-6, "Link %s forward kinematics mismatch, max error: %.3e" % (joint.name, np.max(error))

    print("\n Batch Torch based forward kinematics is working appropriately\n")

