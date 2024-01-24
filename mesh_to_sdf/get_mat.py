from mesh_to_sdf import sample_sdf_near_surface
from scipy.io import savemat
import os

import trimesh
import pyrender
import numpy as np

hand_names = ["ShadowHand", "AllegroHand", "RutgersHand", "SchunkHand", "ManoHand", "RobonautHand2"]
hand_names = ["RobonautHand2"]

def show_points(points, colors):
    scene = pyrender.Scene()
    points = points[:, :3]
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

for hand_name in hand_names:
    mesh = trimesh.load('files/%s.obj'%(hand_name))
    p_sdf, resized_mesh = sample_sdf_near_surface(mesh, surface_point_method='sample', sign_method='normal', number_of_points=500000)
    indices = p_sdf[2]
    label_list = np.load("files/%s.npy"%(hand_name))
    labels = label_list[indices]
    color_list = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
    ])
    p_sdf = np.concatenate([p_sdf[0], np.expand_dims(p_sdf[1], axis=1)], axis=1)
    # sdf = p_sdf[:, 3]
    # colors = np.zeros(points.shape)
    # colors[sdf < 0, 2] = 1
    # colors[sdf > 0, 0] = 1
    colors = color_list[(labels//10).astype(np.int32)]
    show_points(p_sdf, colors)

    # Save free space points
    if not os.path.exists("free_space_pts"):
        os.makedirs("free_space_pts")
    savemat("free_space_pts/"+hand_name+".mat", {"p_sdf":p_sdf, "labels":labels})

    sample_points, sample_idxes =trimesh.sample.sample_surface_even(resized_mesh, 500000)
    sample_normals = mesh.face_normals[sample_idxes]
    labels_surface = label_list[sample_idxes]
    colors_surface = color_list[(labels_surface//10).astype(np.int32)]
    p = np.concatenate([sample_points, sample_normals], axis=1)
    show_points(p, colors_surface)

    # Save on surface points
    if not os.path.exists("surface_pts_n_normal"):
        os.makedirs("surface_pts_n_normal")
    savemat("surface_pts_n_normal/"+hand_name+".mat", {"p": p, "labels":labels_surface})

    print(hand_name, p_sdf.shape, p.shape)
    resized_mesh.export('files/%s_resized.obj'%(hand_name))