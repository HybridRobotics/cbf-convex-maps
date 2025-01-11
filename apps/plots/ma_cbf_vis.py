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
_QUADROTOR_MATERIAL_A = gm.MeshLambertMaterial(
    color=0x00FF00, wireframe=False, opacity=1, reflectivity=0.5
)
_QUADROTOR_MATERIAL_B = gm.MeshLambertMaterial(
    color=0xFF0000, wireframe=False, opacity=1, reflectivity=0.5
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
    vis["quad"].set_object(_QUADROTOR_OBJ, _QUADROTOR_MATERIAL_B)
    vis["quad"].set_transform(T)
    x = np.hstack([np.zeros((6,)), np.identity(3).reshape((-1,))])
    x[3] = 1.0

    vd, fd = _QUADROTOR_DOWNWASH_MESH
    vc, fc = mesh_from_implicit_function(
        lambda Z: qcorridor_implicit_function(Z, x),
        dz=0.1,
        z_lim=[[-2.0, 2.0], [-1.5, 1.5], [-1.5, 1.5]],
    )
    set = 1

    if set == 1:
        # Downwash set.
        downwash_mesh = gm.TriangularMeshGeometry(vd, fd)
        vis["downwash"].set_object(downwash_mesh, _MESH_MATERIAL)
        vis["downwash"].set_transform(T)
        # Corridor set.
        corridor_mesh = gm.TriangularMeshGeometry(vc, fc)
        vis["corridor"].set_object(corridor_mesh, _MESH_MATERIAL)
    elif set == 2:
        vd = vd @ T[:3, :3].T
        v, f = minkowski_sum_mesh_from_vertices(vd, vc)
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

    # Draw obstacle polytopes.
    A_obs, b_obs = [], []
    # Obstacle 1: dodecahedron.
    a1 = 0.52573
    a2 = 0.85065
    a3 = 1.37638
    # fmt: off
    A_obs.append(
        np.array(
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
    )
    # fmt: on
    b_obs.append(a3 * np.ones((12,)))

    num_obs = len(A_obs)
    for i, A, b in zip(range(num_obs), A_obs, b_obs):
        vertices, faces = vertex_face_from_halfspace_rep(A, b)
        f = wavefront_virtual_file(vertices, faces)
        mesh = gm.ObjMeshGeometry.from_stream(f)
        vis["obstacle-" + str(i)].set_object(mesh, _OBSTACLE_MATERIAL)


def _add_quadrotor(vis: Visualizer, x: np.ndarray, name: str, team: int) -> None:
    if team == 0:
        material = _QUADROTOR_MATERIAL_A
    elif team == 1:
        material = _QUADROTOR_MATERIAL_B
    else:
        RuntimeError("Team show be 0 or 1!")
    vis[name]["quad"].set_object(_QUADROTOR_OBJ, material)
    v, f = _QUADROTOR_SHAPE_MESH
    mesh = gm.TriangularMeshGeometry(v, f)
    vis[name]["shape"].set_object(mesh, _MESH_MATERIAL)

    p = x[:3]
    R = x[-9:].reshape((3, 3))
    T = tf.translation_matrix(p)
    T[:3, :3] = R
    vis[name].set_transform(T)


def _snapshot(log: dict, vis: Visualizer, timestamps: np.ndarray) -> None:
    assert all(timestamps >= 0) and all(timestamps <= 1)
    Nlog = len(log["t_seq"])
    x = log["x"]

    _add_obstacle(vis, log)

    for i in range(log["num_sys"]):
        xi = x[15 * i : 15 * (i + 1), :]

        # Create a 3D line plot
        # Nf = max(0, min(Nlog - 1, int(timestamps[-1] * Nlog)))
        # vertices = x[:3, :Nf:10]
        sp = 10
        vertices = xi[:3, ::sp]
        trajectory = gm.Line(
            gm.PointsGeometry(vertices),
            gm.MeshBasicMaterial(color=0xFF0000, transparency=False, opacity=1),
        )
        vis["trajectory-" + str(i)].set_object(trajectory)

        for j, f in enumerate(timestamps):
            idx = max(0, min(Nlog - 1, int(f * Nlog)))
            xit = x[:, idx]
            name = "robot-" + str(i) + "-" + str(j)
            _add_quadrotor(vis, xit, name, i % 2)


if __name__ == "__main__":
    # Read logs.
    relative_path = "/../../build"

    filename = _DIR_PATH + relative_path + "/ma_cbf_data.csv"
    log = read_ma_cbf_logs(filename)

    # Meshcat visualizer.
    vis = meshcat.Visualizer()
    vis.open()

    # Snapshots.
    timestamps = np.array([0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0])
    _snapshot(log, vis, timestamps)
    # _safe_region_snapshot(vis)
