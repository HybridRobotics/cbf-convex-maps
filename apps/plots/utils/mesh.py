import io
from typing import Callable, List

import meshcat
import meshcat.geometry as gm
import numpy as np
import polytope as pc
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull
from scipy.special import logsumexp
from skimage import measure


def faces_from_vertex_rep(vertices: np.ndarray) -> np.ndarray:
    """Returns the face indices of a polytope from its vertex representation.
    Inputs:
    vertices: (n, 3) ndarray of vertices in 3D.
    Outputs:
    faces:    (n, 3) List of face indices.
    """
    assert vertices.shape[1] == 3
    hull = ConvexHull(vertices)
    return hull.simplices


def normals_from_vertex_face(
    vertices: np.ndarray, faces: List[List[int]]
) -> List[List[float]]:
    """Returns the normals given the vertices and face indices.
    Inputs:
    vertices: (n, 3) ndarray of vertices in 3D.
    faces:    (n, 3) List of face indices.
    Outputs:
    normals:  (n, 3) List of normals (in order of faces, with CCW order).
    """
    assert vertices.shape[1] == 3
    assert len(faces[0]) == 3
    normals = []
    for f in faces:
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n)
        # if (center is not None) and (n.dot(v0 - center) < 0):
        #     n = -n
        normals.append(n.tolist())
    return normals


def wavefront_virtual_file(
    vertices: np.ndarray, faces: List[List[int]], normals: List[List[float]] = None
) -> io.StringIO:
    """Returns a virtual .obj file for the meh (use for small meshes).
    Inputs:
    vertices: (n, 3) ndarray of vertices in 3D.
    faces:    (n, 3) List of face indices (0 indexed).
    normals:  (n, 3) List of normals (in order of faces).
    Outputs:
    f:        Virtual .obj file.
    """
    filestr = ""
    faces = faces + 1  # Wavefront format is 1 indexed.

    # Enter vertices.
    for v in vertices:
        filestr += f"v {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f}\n"
    filestr += "\n"

    if normals is None:
        # Enter faces.
        for f in faces:
            filestr += f"f {f[0]} {f[1]} {f[2]}\n"
    else:
        # Enter normals.
        for vn in normals:
            filestr += f"vn {vn[0]:5.3f} {vn[1]:5.3f} {vn[2]:5.3f}\n"
        # Enter faces.
        filestr += "\n"
        for i, f in enumerate(faces):
            i_ = i + 1
            filestr += f"f " + f"{f[0]}//{i_} " + f"{f[1]}//{i_} " + f"{f[2]}//{i_}\n"

    return io.StringIO(filestr)


def vertex_face_from_halfspace_rep(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, List[List[int]]]:
    """Returns the vertices and face indices of a polytope from its halfspace representation.
    Inputs:
    A:        (m, 3) ndarray of halfspace normals,
    b:        (m,) ndarray of halfspace intercepts.
    Outputs:
    vertices: (n, 3) ndarray of vertices,
    faces:    (n, 3) List of face indices.
    """
    assert A.shape[1] == 3
    assert b.ndim == 1
    assert A.shape[0] == b.shape[0]
    p = pc.Polytope(A, b)
    p = pc.reduce(p)
    vertices = pc.extreme(p)
    faces = faces_from_vertex_rep(vertices)
    return (vertices, faces)


def qshape_implicit_function(Z: List[np.ndarray]) -> np.ndarray:
    _pow = 2.5
    _coeff = [1.0, 1.0, 0.4, 0.3]

    f = -np.power(_coeff[3], _pow)
    for i in range(3):
        f += np.power(np.abs(Z[i]) / _coeff[i], _pow)
    return f


def quncertainty_implicit_function(Z: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    _Q = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    _coeff = [1.0, 0.5, 1.0, 0.0]

    level = _coeff[0] + 2 / np.pi * _coeff[1] * np.arctan(_coeff[2] * x[2] + _coeff[3])
    f = -level
    for i in range(3):
        for j in range(3):
            f += _Q[i, j] * Z[i] * Z[j]
    return f


def qdownwash_implicit_function(Z: List[np.ndarray]) -> np.ndarray:
    _A = np.array(
        [
            [4.0, 0.0, 2.0],
            [0.0, 4.0, 2.0],
            [-4.0, 0.0, 2.0],
            [0.0, -4.0, 2.0],
            [0.0, 0.0, -1.5],
        ]
    )
    _b = np.zeros((5,))
    level = 1.5

    Zp = []
    for i in range(_A.shape[0]):
        Zp.append(_A[i, 0] * Z[0] + _A[i, 1] * Z[1] + _A[i, 2] * Z[2] - _b[i])
    f = logsumexp(Zp, axis=0) - level
    return f


def qcorridor_implicit_function(Z: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    _stop_time = 1.0
    _orientation_cost = 0.5
    _max_vel = 1.0
    _eps = 0.2
    _ecc = 5
    _min = 0.1
    _max_q = 0.1 * _max_vel * (_stop_time + 2 * _orientation_cost)

    v = x[3:6]
    R = x[-9:].reshape((3, 3))
    v_norm = np.sqrt((_eps * _max_vel) ** 2 + v.dot(v))
    nv = v / v_norm
    corridor_time = _stop_time + _orientation_cost * (1 + R[:, 2].dot(nv))
    q = v * corridor_time
    q_norm = np.sqrt((_eps * _max_q) ** 2 + q.dot(q))
    nq = q / q_norm
    Q = _ecc**2 * np.identity(3) - (_ecc**2 - 1) * nq.reshape((3, 1)) * nq
    level = (_min + q_norm / 2) ** 2

    f = -level
    for i in range(3):
        for j in range(3):
            f += Q[i, j] * (Z[i] - q[i] / 2) * (Z[j] - q[j] / 2)
    return f


def mesh_from_implicit_function(
    function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    dz: float,
    z_lim: List[List[float]],
) -> tuple[np.ndarray, np.ndarray]:
    Z0, Z1, Z2 = np.mgrid[
        z_lim[0][0] : z_lim[0][1] : dz,
        z_lim[1][0] : z_lim[1][1] : dz,
        z_lim[2][0] : z_lim[2][1] : dz,
    ]
    values = function([Z0, Z1, Z2])

    vertices, faces, _, _ = measure.marching_cubes(values, level=0, spacing=[dz] * 3)
    vertices -= np.array(
        [
            (z_lim[0][1] - z_lim[0][0]) / 2,
            (z_lim[1][1] - z_lim[1][0]) / 2,
            (z_lim[2][1] - z_lim[2][0]) / 2,
        ]
    )
    return (vertices, faces)


def minkowski_sum_mesh_from_vertices(
    vertices1: np.ndarray, vertices2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    assert (vertices1.shape[1] == 3) and (vertices2.shape[1] == 3)
    assert (vertices1.shape[0] <= 250) and (vertices2.shape[0] <= 200)

    poly1 = pc.qhull(vertices1[::3, :])
    poly2 = pc.qhull(vertices2[::3, :])
    A = block_diag(poly1.A, poly2.A)
    n = poly1.A.shape
    A[: n[0], n[1] :] = -poly1.A
    b = np.hstack([poly1.b, poly2.b])
    poly = pc.Polytope(A, b)
    sum = poly.project([1, 2, 3])
    vertices = pc.extreme(sum)
    return vertices, faces_from_vertex_rep(vertices)


def _dodecahedron_vertices() -> None:
    golden = (1 + 5**0.5) / 2
    vertices = []
    for i in range(8):
        vertices.append(
            [(-1) ** (i % 2), (-1) ** ((i // 2) % 2), (-1) ** ((i // 4) % 2)]
        )
    for i in range(4):
        a = (-1) ** (i % 2) * golden
        b = (-1) ** ((i // 2) % 2) / golden
        vertices.append([0, a, b])
        vertices.append([b, 0, a])
        vertices.append([a, b, 0])
    vertices = np.array(vertices)

    if False:
        p = pc.qhull(vertices)
        p = pc.reduce(p)
        print(f"constraints:")
        for i in range(p.A.shape[0]):
            for j in range(3):
                print(f"{p.A[i,j]:8.5f} ", end="")
            print(f"| {p.b[i]:8.5f}")

    if False:
        vis = meshcat.Visualizer()
        vis.open()
        material = gm.MeshLambertMaterial(
            color=0xFFFFFF, wireframe=False, opacity=1, reflectivity=0.5
        )
        faces = faces_from_vertex_rep(vertices)
        normals = normals_from_vertex_face(vertices, faces)
        # f = wavefront_virtual_file(vertices, faces, normals)
        f = wavefront_virtual_file(vertices, faces)

        mesh = gm.ObjMeshGeometry.from_stream(f)
        vis["dodecahedron"].set_object(mesh, material)

    return vertices


def _visualize_sets() -> None:
    p = np.array([0.0, 0.0, 0.0])
    v = 1.0 * np.array([1.0, 0.0, 0.0])
    R = np.identity(3)
    x = np.hstack([p, v, R.reshape((-1,))])

    vis = meshcat.Visualizer()
    vis.open()
    mesh_material = gm.MeshBasicMaterial(
        color=0xFF22DD, wireframe=True, linewidth=2, opacity=1
    )

    # Quadrotor shape
    verts_s, faces_s = mesh_from_implicit_function(
        qshape_implicit_function,
        dz=0.075,
        z_lim=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    )
    print(f"#vertices (qshape) = {verts_s.shape[0]}")
    mesh_s = gm.TriangularMeshGeometry(verts_s, faces_s)
    # vis["qshape"].set_object(mesh_s, mesh_material)

    # Quadrotor uncertainty set
    verts_u, faces_u = mesh_from_implicit_function(
        lambda Z: quncertainty_implicit_function(Z, x),
        dz=0.25,
        z_lim=[[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]],
    )
    print(f"#vertices (quncertainty) = {verts_u.shape[0]}")
    mesh_u = gm.TriangularMeshGeometry(verts_u, faces_u)
    # vis["quncertainty"].set_object(mesh_u, mesh_material)

    # Shape + uncertainty set
    verts_su, faces_su = minkowski_sum_mesh_from_vertices(verts_s, verts_u)
    mesh_su = gm.TriangularMeshGeometry(verts_su, faces_su)
    # vis["q(shape+uncertainty)"].set_object(mesh_su, mesh_material)

    # Quadrotor downwash
    verts_d, faces_d = mesh_from_implicit_function(
        qdownwash_implicit_function,
        dz=0.12,
        z_lim=[[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]],
    )
    print(f"#vertices (qdownwash) = {verts_d.shape[0]}")
    mesh_d = gm.TriangularMeshGeometry(verts_d, faces_d)
    # vis["qdownwash"].set_object(mesh_d, mesh_material)

    # Quadrotor braking corridor
    verts_c, faces_c = mesh_from_implicit_function(
        lambda Z: qcorridor_implicit_function(Z, x),
        dz=0.1,
        z_lim=[[-2.0, 2.0], [-1.5, 1.5], [-1.5, 1.5]],
    )
    print(f"#vertices (qcorridor) = {verts_c.shape[0]}")
    mesh_c = gm.TriangularMeshGeometry(verts_c, faces_c)
    # vis["qcorridor"].set_object(mesh_c, mesh_material)

    # Downwash + corridor set
    verts_dc, faces_dc = minkowski_sum_mesh_from_vertices(verts_d, verts_c)
    mesh_dc = gm.TriangularMeshGeometry(verts_dc, faces_dc)
    # vis["q(downwash+corridor)"].set_object(mesh_dc, mesh_material)


if __name__ == "__main__":
    _visualize_sets()
    _dodecahedron_vertices()
