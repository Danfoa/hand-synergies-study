"""
MIT License

Copyright (c) 2020 Daniel Ordoñez

This code is largely based on Matthew Matl animate function for the URDF class in the python package urdfpy

Copyright (c) 2019 Matthew Matl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import numpy as np
import pyrender
from urdfpy import URDF
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc

rc('animation', html='html5')

def setup_animation_scene(real_traj, pred_traj, urdf_path, hand_offset, loop_time, real_color, pred_color_cmap,
                          background_color, reverse=False):
    pred_trajectories = pred_traj if isinstance(pred_traj, list) else [pred_traj]
    real_hand = URDF.load(urdf_path)
    pred_hands = []

    for i, traj in enumerate(pred_trajectories):
        pred_hands.append(URDF.load(urdf_path))
        pred_origin = pred_hands[i].joint_map['palm_joint_abduction'].origin[:3, 3]
        pred_hands[i].joint_map['palm_joint_abduction'].origin[:3, 3] = pred_origin + np.array(
            [hand_offset * (i + 1), 0.0, 0.0])

    traj_len = None  # Length of the trajectory in steps

    # If it is specified, parse it and extract the trajectory length.
    for pred_traj in pred_trajectories:
        if isinstance(real_traj, dict) and isinstance(pred_traj, dict):
            if len(real_traj) > 0:
                for joint_name in real_traj:
                    if len(real_traj[joint_name]) != len(pred_traj[joint_name]):
                        raise ValueError('Real and pred trajectories [%d] must be same length' % joint_name)
                    elif traj_len is None:
                        traj_len = len(real_traj[joint_name])
                    elif traj_len != len(real_traj[joint_name]):
                        raise ValueError('All joint trajectories must be same length')
        else:
            raise TypeError('Invalid type for trajectories real[%s], pred[%s]' % (type(real_traj), type(pred_traj)))

    # Create an array of times that loops from 0 to 1 and back to 0
    fps = 30.0

    # Create the new interpolated trajectory
    if reverse:
        n_steps = int(loop_time * 2 * fps)
        times = np.linspace(0.0, 1.0, n_steps)
        times = np.hstack((times, np.flip(times)))
        # Create bin edges in the range [0, 1] for each trajectory step
        bins = np.arange(traj_len) / (float(traj_len) - 1.0)

        # Compute alphas for each time
        right_inds = np.digitize(times, bins, right=True)
        right_inds[right_inds == 0] = 1
        alphas = ((bins[right_inds] - times) /
                  (bins[right_inds] - bins[right_inds - 1]))

        for joint_name in real_traj:
            real_traj[joint_name] = (alphas * real_traj[joint_name][right_inds - 1] +
                                     (1.0 - alphas) * real_traj[joint_name][right_inds])
            # Compute all
            for pred_traj in pred_trajectories:
                pred_traj[joint_name] = (alphas * pred_traj[joint_name][right_inds - 1] +
                                         (1.0 - alphas) * pred_traj[joint_name][right_inds])
    else:
        n_steps = int(loop_time * fps)
        times = np.linspace(0.0, 1.0, n_steps)

    # Create the scene
    fk_real = real_hand.visual_trimesh_fk()
    fk_preds = [pred_hand.visual_trimesh_fk() for pred_hand in pred_hands]

    node_map = {}
    scene = pyrender.Scene(bg_color=background_color)
    # Spawn ground truth hand
    for tm_real in fk_real:
        # Little hack to overcome the urdfpy bug of ignoring URDF materials for .stl meshes
        tm_real._visual.face_colors = tm_real._visual.face_colors * 0 + real_color

        # Real hand nodes
        pose = fk_real[tm_real]
        real_mesh = pyrender.Mesh.from_trimesh(tm_real, smooth=False)
        node = scene.add(real_mesh, pose=pose)
        node_map[tm_real] = node

    # Spawn prediction hands
    pred_color = []
    for color_code in np.linspace(0, 1, len(pred_trajectories)):
        pred_color.append(pred_color_cmap(color_code, bytes=True))

    for i, fk_pred in enumerate(fk_preds):
        for tm_pred in fk_pred:
            # Little hack to overcome the urdfpy bug of ignoring URDF materials for .stl meshes
            tm_pred._visual.face_colors = tm_pred._visual.face_colors * 0 + pred_color[i]

            # Pred hand nodes
            pose = fk_pred[tm_pred]
            pred_mesh = pyrender.Mesh.from_trimesh(tm_pred, smooth=False)
            node = scene.add(pred_mesh, pose=pose)
            node_map[tm_pred] = node

    # Get base pose to focus on
    origin = real_hand.link_fk(links=[real_hand.base_link])[real_hand.base_link]

    return scene, origin, node_map, real_hand, pred_hands, pred_trajectories, times, fps

def prediction_animation(real_traj, pred_traj, loop_time,
                         urdf_path='robots/right_hand_relative.urdf',
                         real_color=np.array([71, 107, 107, 255]),
                         pred_color_cmap=matplotlib.cm.get_cmap('tab10'),
                         background_color=np.array([1.0, 1.0, 1.0]),
                         title="Hand Motion Prediction",
                         hand_offset=0.2,
                         reverse=True):

    scene, origin, node_map, real_hand, pred_hands, pred_trajectories, times, fps = setup_animation_scene(
        real_traj=real_traj, pred_traj=pred_traj,
        hand_offset=hand_offset, loop_time=loop_time, urdf_path=urdf_path, real_color=real_color,
        pred_color_cmap=pred_color_cmap, background_color=background_color, reverse=reverse)

    # Pop the visualizer asynchronously
    v = pyrender.Viewer(scene,
                        run_in_thread=True,
                        use_raymond_lighting=True,
                        window_title=title,
                        use_perspective_cam=False,
                        view_center=origin[:3, 3] + np.array([(hand_offset * (len(pred_trajectories) + 1)) / 2, 0, 0.04]))

    # Now, run our loop
    i = 0
    while v.is_active:
        real_cfg = {k: real_traj[k][i] for k in real_traj}
        pred_cfgs = [{k: pred_traj[k][i] for k in pred_traj} for pred_traj in pred_trajectories]
        i = (i + 1) % len(times)

        fk_real = real_hand.visual_trimesh_fk(cfg=real_cfg)
        fk_preds = [pred_hand.visual_trimesh_fk(cfg=pred_cfg) for pred_hand, pred_cfg in zip(pred_hands, pred_cfgs)]

        v.render_lock.acquire()
        for real_mesh in fk_real:
            real_pose = fk_real[real_mesh]
            node_map[real_mesh].matrix = real_pose
        for fk_pred in fk_preds:
            for pred_mesh in fk_pred:
                pred_pose = fk_pred[pred_mesh]
                node_map[pred_mesh].matrix = pred_pose
        v.render_lock.release()

        time.sleep(1.0 / fps)


def fixed_prediction_animation(real_traj, pred_traj, loop_time,
                               urdf_path='robots/right_hand_relative.urdf',
                               real_color=np.array([71, 107, 107, 255]),
                               pred_color_cmap=matplotlib.cm.get_cmap('tab10'),
                               background_color=np.array([1.0, 1.0, 1.0]),
                               hand_offset=0.2,
                               title="Hand Motion Prediction",
                               reverse=True,
                               show=False):
    scene, origin, node_map, real_hand, pred_hands, pred_trajectories, times, fps = setup_animation_scene(
        real_traj=real_traj, pred_traj=pred_traj,
        hand_offset=hand_offset, loop_time=loop_time, urdf_path=urdf_path, real_color=real_color,
        pred_color_cmap=pred_color_cmap, background_color=background_color, reverse=reverse)

    # clear_output()

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, hand_offset / 2],
        [0.0, 0.0, -1.0, -0.3],
        [0.0, 1.0, 0.0, 0.10],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5)
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(1200, 800)

    rgb_sequence, depth_sequence = [], []

    fig = plt.figure(figsize=(10, 8))
    plt.title(title)

    color, _ = r.render(scene)
    rgb_sequence.append(color)

    for i in tqdm(range(len(times)), position=0, leave=True, desc="Rendering"):
        real_cfg = {k: real_traj[k][i] for k in real_traj}
        pred_cfgs = [{k: pred_traj[k][i] for k in pred_traj} for pred_traj in pred_trajectories]
        # i = (i + 1) % len(times)

        fk_real = real_hand.visual_trimesh_fk(cfg=real_cfg)
        fk_preds = [pred_hand.visual_trimesh_fk(cfg=pred_cfg) for pred_hand, pred_cfg in zip(pred_hands, pred_cfgs)]

        for real_mesh in fk_real:
            real_pose = fk_real[real_mesh]
            node_map[real_mesh].matrix = real_pose
        for fk_pred in fk_preds:
            for pred_mesh in fk_pred:
                pred_pose = fk_pred[pred_mesh]
                node_map[pred_mesh].matrix = pred_pose

        color, _ = r.render(scene)
        rgb_sequence.append(color)

    r.delete()

    artists = []
    for img in tqdm(rgb_sequence, position=0, leave=True, desc="Preparing PLT Animation"):
        img_color = plt.imshow(img, animated=True)
        artists.append([img_color])

    # Create PLT animation
    ani = animation.ArtistAnimation(fig, artists, interval=1 / fps * 1000, blit=True, repeat_delay=500)
    plt.axis('off')
    if show:
        plt.show()
    else:
        plt.close()
    return ani
