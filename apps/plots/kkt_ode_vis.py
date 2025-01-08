import os

import meshcat
import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
from meshcat import Visualizer
from utils.datalog import read_kkt_ode_logs
from utils.mesh import (
    mesh_from_implicit_function,
    minkowski_sum_mesh_from_vertices,
    qshape_implicit_function,
    quncertainty_implicit_function,
    vertex_face_from_halfspace_rep,
    wavefront_virtual_file,
)

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

_QUADROTOR_OBJ_PATH = _DIR_PATH + "/quadrotor.obj"
_QUADROTOR_OBJ = gm.ObjMeshGeometry.from_file(_QUADROTOR_OBJ_PATH)
_QUADROTOR_SHAPE_MESH = mesh_from_implicit_function(
    qshape_implicit_function,
    dz=0.075,
    z_lim=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
)
_QUADROTOR_UNCERTAINTY_MESH = mesh_from_implicit_function(
    lambda Z: quncertainty_implicit_function(
        Z,
        np.hstack([np.zeros((6,)), np.identity(3).reshape((-1,))]),
    ),
    dz=0.25,
    z_lim=[[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]],
)

_OBSTACLE_MATERIAL = gm.MeshLambertMaterial(
    color=0xFFFFFF, wireframe=False, opacity=0.75, reflectivity=0.5
)
_QUADROTOR_MATERIAL = gm.MeshLambertMaterial(
    color=0xAAAAAA, wireframe=False, opacity=1, reflectivity=0.5
)
_MESH_MATERIAL = gm.MeshBasicMaterial(
    color=0xFF22DD, wireframe=True, linewidth=1, opacity=1
)


def _minkowski_sum_snapshot(vis: Visualizer) -> None:
    # Remove grid.
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)
    # Change background color.
    vis["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
    vis["/Background"].set_property("bottom_color", [1.0, 1.0, 1.0])

    vis["quad"].set_object(_QUADROTOR_OBJ, _QUADROTOR_MATERIAL)
    set = 1

    if set == 1:
        v, f = _QUADROTOR_SHAPE_MESH
        shape_mesh = gm.TriangularMeshGeometry(v, f)
        vis["shape"].set_object(shape_mesh, _MESH_MATERIAL)
    elif set == 2:
        v, f = _QUADROTOR_UNCERTAINTY_MESH
        uncertainty_mesh = gm.TriangularMeshGeometry(v, f)
        vis["uncertainty"].set_object(uncertainty_mesh, _MESH_MATERIAL)
    elif set == 3:
        vs, _ = _QUADROTOR_SHAPE_MESH
        vu, _ = _QUADROTOR_UNCERTAINTY_MESH
        v, f = minkowski_sum_mesh_from_vertices(vs, vu)
        safe_mesh = gm.TriangularMeshGeometry(v, f)
        vis["safe"].set_object(safe_mesh, _MESH_MATERIAL)

    vis.set_cam_pos(np.array([0.0, 0.0, 2.0]))
    vis.set_cam_target(np.zeros((3,)))


def _add_obstacle(vis: Visualizer, log: dict) -> None:
    # Remove grid.
    vis["/Grid"].set_property("visible", False)
    # Change background color.
    # vis["/Background"].set_property("top_color", [0.9, 0.9, 0.9])
    vis["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])

    # Draw obstacle polytope.
    A, b = log["A"], log["b"]
    vertices, faces = vertex_face_from_halfspace_rep(A, b)
    f = wavefront_virtual_file(vertices, faces)
    mesh = gm.ObjMeshGeometry.from_stream(f)
    vis["obstacle"].set_object(mesh, _OBSTACLE_MATERIAL)


def _add_quadrotor(vis: Visualizer, x: np.ndarray, name: str) -> None:
    vis[name]["quad"].set_object(
        _QUADROTOR_OBJ,
        _QUADROTOR_MATERIAL,
    )
    x_ = np.hstack([np.zeros((6,)), np.identity(3).reshape((-1,))])
    Z_ = [np.array([0.0])] * 3
    level0 = quncertainty_implicit_function(Z_, x_)
    x_[2] = x[2]
    level = quncertainty_implicit_function(Z_, x_)
    scale = np.sqrt(level / level0)

    vertices_shape, _ = _QUADROTOR_SHAPE_MESH
    vertices_uncertainty, _ = _QUADROTOR_UNCERTAINTY_MESH
    vertices_uncertainty = vertices_uncertainty * scale
    vf = minkowski_sum_mesh_from_vertices(vertices_shape, vertices_uncertainty)
    safe_mesh = gm.TriangularMeshGeometry(*vf)
    vis[name]["safe"].set_object(safe_mesh, _MESH_MATERIAL)

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

    # Create a 3D line plot
    Nf = max(0, min(Nlog - 1, int(timestamps[-1] * Nlog)))
    # vertices = x[:3, :Nf:10]
    vertices = x[:3, ::10]
    trajectory = gm.Line(
        gm.PointsGeometry(vertices),
        gm.MeshBasicMaterial(color=0xFF0000, transparency=False, opacity=1),
    )
    vis["trajectory"].set_object(trajectory)

    for i, f in enumerate(timestamps):
        idx = max(0, min(Nlog - 1, int(f * Nlog)))
        xi = x[:, idx]
        name = "robot-" + str(i)
        _add_quadrotor(vis, xi, name)


if __name__ == "__main__":
    # Read logs.
    relative_path = "/../../build"
    type = 0

    filename = _DIR_PATH + relative_path + "/kkt_ode_data_" + str(type) + ".csv"
    log = read_kkt_ode_logs(filename)

    # Meshcat visualizer.
    vis = meshcat.Visualizer()
    vis.open()

    # Snapshots.
    timestamps = np.array([0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0])
    _snapshot(log, vis, timestamps)
    # _minkowski_sum_snapshot(vis)
