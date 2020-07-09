import numpy as np

from utils.data_loader_kine_adl import DATABASE_PATH, RightHand, load_subject_data
from utils.data_visualizer_kine_adl import RightHandJointNames, create_joint_state_msg

from utils.pyrender_visualization import animate_prediction

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

    animate_prediction(real_traj=cfg, pred_traj=pred_cfg, loop_time=loop_time,
                       real_color=np.array([58, 79, 87, 255]),
                       pred_color=np.array([117, 46, 18, 255]),
                       run_in_thread=True
                       )

    # robot = URDF.load('robots/right_hand_relative.urdf')

    # default_cfg = {joint: np.linspace(0, 0.5, 50) for joint in robot.joints}

    # robot.animate(cfg_trajectory=cfg, loop_time=0.01*2*df.shape[0])
