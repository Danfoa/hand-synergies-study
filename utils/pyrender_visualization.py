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
from urdfpy import URDF

def animate_prediction(real_traj, pred_traj, loop_time,
                       real_color=np.array([71, 107, 107, 255]),
                       pred_color=np.array([209, 224, 224, 255]),
                       background_color=np.array([1.0, 1.0, 1.0]),
                       pred_hand_offset=0.15,
                       run_in_thread=True):
    import pyrender

    real_hand = URDF.load('robots/right_hand_relative.urdf')
    pred_hand = URDF.load('robots/right_hand_relative.urdf')

    pred_origin = pred_hand.joint_map['palm_joint_abduction'].origin[:3, 3]
    pred_hand.joint_map['palm_joint_abduction'].origin[:3, 3] = pred_origin + np.array([pred_hand_offset, 0.0, 0.0])
    ct = real_traj

    traj_len = None  # Length of the trajectory in steps

    # If it is specified, parse it and extract the trajectory length.
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
    n_steps = int(loop_time * fps / 2.0)
    times = np.linspace(0.0, 1.0, n_steps)
    times = np.hstack((times, np.flip(times)))

    # Create bin edges in the range [0, 1] for each trajectory step
    bins = np.arange(traj_len) / (float(traj_len) - 1.0)

    # Compute alphas for each time
    right_inds = np.digitize(times, bins, right=True)
    right_inds[right_inds == 0] = 1
    alphas = ((bins[right_inds] - times) /
              (bins[right_inds] - bins[right_inds - 1]))

    # Create the new interpolated trajectory
    new_real_traj, new_pred_traj = real_traj, pred_traj

    for joint_name in real_traj:
        new_real_traj[joint_name] = (alphas * real_traj[joint_name][right_inds - 1] +
                            (1.0 - alphas) * real_traj[joint_name][right_inds])
        new_pred_traj[joint_name] = (alphas * pred_traj[joint_name][right_inds - 1] +
                            (1.0 - alphas) * pred_traj[joint_name][right_inds])

    # Create the scene
    fk_real = real_hand.visual_trimesh_fk()
    fk_pred = pred_hand.visual_trimesh_fk()

    node_map = {}
    scene = pyrender.Scene(bg_color=background_color)
    for tm_real, tm_pred in zip(fk_real, fk_pred):
        # Little hack to overcome the urdfpy bug of ignoring URDF materials for .stl meshes
        tm_real._visual.face_colors = tm_real._visual.face_colors * 0 + real_color
        tm_pred._visual.face_colors = tm_pred._visual.face_colors * 0 + pred_color

        # Real hand nodes
        pose = fk_real[tm_real]
        real_mesh = pyrender.Mesh.from_trimesh(tm_real, smooth=False)
        node = scene.add(real_mesh, pose=pose)
        node_map[tm_real] = node

        # Pred hand nodes
        pose = fk_pred[tm_pred]
        pred_mesh = pyrender.Mesh.from_trimesh(tm_pred, smooth=False)
        node = scene.add(pred_mesh, pose=pose)
        node_map[tm_pred] = node

    # Get base pose to focus on
    blp = real_hand.link_fk(links=[real_hand.base_link])[real_hand.base_link]

    # Pop the visualizer asynchronously
    v = pyrender.Viewer(scene,
                        run_in_thread=run_in_thread,
                        use_raymond_lighting=True,
                        view_center=blp[:3, 3] + np.array([pred_hand_offset/2, 0, 0.05]))

    # Now, run our loop
    i = 0
    while v.is_active:
        real_cfg = {k: new_real_traj[k][i] for k in new_real_traj}
        pred_cfg = {k: new_pred_traj[k][i] for k in new_pred_traj}
        i = (i + 1) % len(times)

        fk_real = real_hand.visual_trimesh_fk(cfg=real_cfg)
        fk_pred = pred_hand.visual_trimesh_fk(cfg=pred_cfg)

        v.render_lock.acquire()
        for real_mesh, pred_mesh in zip(fk_real, fk_pred):
            real_pose = fk_real[real_mesh]
            node_map[real_mesh].matrix = real_pose

            pred_pose = fk_pred[pred_mesh]
            node_map[pred_mesh].matrix = pred_pose
        v.render_lock.release()

        time.sleep(1.0 / fps)
