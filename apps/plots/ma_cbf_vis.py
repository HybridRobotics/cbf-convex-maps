import os

import meshcat
import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
from meshcat import Visualizer
from utils.datalog import read_ma_cbf_logs
from utils.mesh import (
    mesh_from_implicit_function,
    minkowski_sum_mesh_from_vertices,
    qcorridor_implicit_function,
    qdownwash_implicit_function,
    qshape_implicit_function,
    vertex_face_from_halfspace_rep,
    wavefront_virtual_file,
)

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

_QUADROTOR_OBJ_PATH = _DIR_PATH + "/quadrotor.obj"
_QUADROTOR_OBJ = gm.ObjMeshGeometry.from_file(_QUADROTOR_OBJ_PATH)
_QUADROTOR_SHAPE_MESH = mesh_from_implicit_function(
    qshape_implicit_function, dz=0.075, z_lim=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
)
_QUADROTOR_DOWNWASH_MESH = mesh_from_implicit_function(
    qdownwash_implicit_function, dz=0.12, z_lim=[[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]]
)

_OBSTACLE_MATERIAL = gm.MeshLambertMaterial(
    color=0xFFFFFF, wireframe=False, opacity=0.75, reflectivity=0.5
)
_QUADROTOR_MATERIAL = gm.MeshLambertMaterial(
    color=0xAAAAAA, wireframe=False, opacity=1, reflectivity=0.5
)
_QUADROTOR_MATERIAL_VIS = gm.MeshLambertMaterial(
    color=0x00FF00, wireframe=False, opacity=1, reflectivity=0.5
)
_MESH_MATERIAL = gm.MeshBasicMaterial(
    color=0xFF22DD, wireframe=True, linewidth=1, opacity=1
)


def _safe_region_snapshot(vis: Visualizer) -> None:
    # Remove grid.
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)
    # Change background color.
    vis["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
    vis["/Background"].set_property("bottom_color", [1.0, 1.0, 1.0])

    T = tf.rotation_matrix(np.pi / 4.0, [0, 1, 0])
    vis["quad"].set_object(_QUADROTOR_OBJ, _QUADROTOR_MATERIAL)
    vis["quad"].set_transform(T)
    x = np.hstack([np.zeros((6,)), np.identity(3).reshape((-1,))])
    x[3] = 0.5

    vs, fs = _QUADROTOR_SHAPE_MESH
    vc, fc = mesh_from_implicit_function(
        lambda Z: qcorridor_implicit_function(Z, x),
        dz=0.1,
        z_lim=[[-2.0, 2.0], [-1.5, 1.5], [-1.5, 1.5]],
    )
    set = 2

    if set == 1:
        # Shape set.
        shape_mesh = gm.TriangularMeshGeometry(vs, fs)
        vis["shape"].set_object(shape_mesh, _MESH_MATERIAL)
        vis["shape"].set_transform(T)
        # Corridor set.
        corridor_mesh = gm.TriangularMeshGeometry(vc, fc)
        vis["corridor"].set_object(corridor_mesh, _MESH_MATERIAL)
    elif set == 2:
        vs = vs @ T[:3, :3].T
        v, f = minkowski_sum_mesh_from_vertices(vs, vc)
        safe_mesh = gm.TriangularMeshGeometry(v, f)
        vis["safe"].set_object(safe_mesh, _MESH_MATERIAL)

    vis.set_cam_pos(np.array([1.0, -2.0, 0.0]))
    vis.set_cam_target(np.array([0.0, 0.0, 0.0]))


def _add_obstacle(vis: Visualizer, log: dict) -> None:
    # Remove grid.
    vis["/Grid"].set_property("visible", False)
    # Change background color.
    # vis["/Background"].set_property("top_color", [0.9, 0.9, 0.9])
    vis["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])

    # Dodecahedron polytope.
    a1 = 0.52573
    a2 = 0.85065
    a3 = 1.37638
    # fmt: off
    A_20 = np.array(
        [
            [a1, a2, 0.0],
            [a1, -a2, 0.0],
            [-a1, a2, 0.0],
            [-a1, -a2, 0.0],
            [0.0, a1, a2],
            [0.0, a1, -a2],
            [0.0, -a1, a2],
            [0.0, -a1, -a2],
            [a2, 0.0, a1],
            [-a2, 0.0, a1],
            [a2, 0.0, -a1],
            [-a2, 0.0, -a1],
        ]
    )
    # fmt: on
    b_20 = a3 * np.ones((12,))

    # Draw obstacle polytopes.
    A_obs, b_obs = [], []
    #   Obstacle 1.
    A_obs.append(A_20)
    p = np.array([-7.0, 1.5, -0.0])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 2.
    A_obs.append(A_20)
    p = np.array([-6.0, -1.5, -2.0])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 3.
    A_obs.append(A_20)
    p = np.array([-3.5, 1.5, -2.5])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 4.
    A_obs.append(A_20)
    p = np.array([-0.0, 1.0, -1.5])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 5.
    A_obs.append(A_20)
    p = np.array([1.0, -1.0, -1.5])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 6.
    A_obs.append(A_20)
    p = np.array([4.0, 1.0, 1.0])
    b_obs.append(b_20 + A_20 @ p)
    #   Obstacle 7.
    A_obs.append(A_20)
    p = np.array([8.0, 0.0, -1.5])
    b_obs.append(b_20 + A_20 @ p)

    num_obs = len(A_obs)
    for i, A, b in zip(range(num_obs), A_obs, b_obs):
        vertices, faces = vertex_face_from_halfspace_rep(A, b)
        f = wavefront_virtual_file(vertices, faces)
        mesh = gm.ObjMeshGeometry.from_stream(f)
        vis["obstacle-" + str(i)].set_object(mesh, _OBSTACLE_MATERIAL)

    # Walls
    l = 12.0  # [m]
    w = 3.0  # [m]
    h = 3.0  # [m]
    vertices = np.array(
        [
            [l, w, h],
            [-l, w, h],
            [-l, -w, h],
            [l, -w, h],
            [l, w, h],
            [l, w, -h],
            [-l, w, -h],
            [-l, -w, -h],
            [l, -w, -h],
            [l, w, -h],
            [-l, w, -h],
            [-l, w, h],
            [-l, -w, h],
            [-l, -w, -h],
            [l, -w, -h],
            [l, -w, h],
        ]
    ).T
    wall = gm.Line(
        gm.PointsGeometry(vertices),
        gm.MeshBasicMaterial(color=0xFF22DD, transparency=False, opacity=1),
    )
    vis["wall"].set_object(wall)


def _add_quadrotor(vis: Visualizer, x: np.ndarray, name: str) -> None:
    p = x[:3]
    R = x[-9:].reshape((3, 3))
    T = tf.translation_matrix(p)
    T[:3, :3] = R
    vis[name]["quad"].set_object(_QUADROTOR_OBJ, _QUADROTOR_MATERIAL_VIS)
    vis[name].set_transform(T)


def _snapshot(log: dict, vis: Visualizer, timestamps: np.ndarray) -> None:
    assert all(timestamps >= 0) and all(timestamps <= 1)
    Nlog = len(log["t_seq"])
    x = log["x"]

    _add_obstacle(vis, log)

    for i in range(log["num_sys"]):
        xi = x[15 * i : 15 * (i + 1), :]

        # Create a 3D line plot
        Tf = 80  # [s]
        Nf = int(Tf / 100.0 * len(log["t_seq"]) - 1)
        sp = 10
        vertices = xi[:3, :Nf:sp]
        trajectory = gm.Line(
            gm.PointsGeometry(vertices),
            gm.MeshBasicMaterial(color=0x0000FF, transparency=False, opacity=1),
        )
        vis["trajectory-" + str(i)].set_object(trajectory)

        for j, f in enumerate(timestamps):
            idx = max(0, min(Nlog - 1, int(f * Nlog)))
            xif = x[:, idx]
            name = "robot-" + str(i) + "-" + str(j)
            _add_quadrotor(vis, xif, name)


if __name__ == "__main__":
    # Read logs.
    relative_path = "/../../build"

    filename = _DIR_PATH + relative_path + "/ma_cbf_data.csv"
    log = read_ma_cbf_logs(filename)

    # Meshcat visualizer.
    vis = meshcat.Visualizer()
    vis.open()

    # Snapshots.
    timestamps = np.arange(0.0, 1.0, 0.05)
    # _snapshot(log, vis, timestamps)
    _safe_region_snapshot(vis)
