import numpy as np

from utils.data_loader_kine_adl import DATABASE_PATH, RightHand, load_subject_data
from utils.data_visualizer_kine_adl import RightHandJointNames, create_joint_state_msg

from utils.pyrender_visualization import prediction_animation, fixed_prediction_animation

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

    cfg = {joint_name: positions[:, i] for i, joint_name in enumerate(joint_names)}

    pred_cfg = {joint_name: positions[:, i] * 1.3 for i, joint_name in enumerate(joint_names)}

    loop_time = 0.01 * 2 * df.shape[0]

    prediction_animation(real_traj=cfg, pred_traj=pred_cfg, loop_time=loop_time,
                         urdf_path='robots/right_hand_relative.urdf',
                         real_color=np.array([71, 107, 107, 255]),
                         pred_color=np.array([209, 224, 224, 255]),
                         background_color=np.array([1.0, 1.0, 1.0]),
                         pred_hand_offset=0.2,
                         )

    # fixed_prediction_animation(real_traj=cfg, pred_traj=pred_cfg, loop_time=loop_time/2,
    #                            urdf_path='robots/right_hand_relative.urdf',
    #                            real_color=np.array([71, 107, 107, 255]),
    #                            pred_color=np.array([209, 224, 224, 255]),
    #                            background_color=np.array([1.0, 1.0, 1.0]),
    #                            pred_hand_offset=0.2,
    #                            title="Hand Motion Prediction",
    #                            show=True)
