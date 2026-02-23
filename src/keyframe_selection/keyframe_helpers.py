import numpy as np
from .keyframe_selec import Keyframe, Edge, MapPoint


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T


def kps_to_xy(kps):
    return np.array([kp.pt for kp in kps], dtype=np.float32)


def get_kf_id_by_frame(slam_map, frame_id: int):
    """
    Retrieve the keyframe ID for a given frame ID from the map.

    Parameters
    ----------
    slam_map : Map
        The SLAM map containing keyframes.
    frame_id : int
        The frame ID to search for.

    Returns
    -------
    int or None
        The keyframe ID if found, otherwise None.
    """
    for kf_id, kf in slam_map.keyframes.items():
        if kf.frame_id == frame_id:
            return kf_id
    return None


def insert_keyframe(
    slam_map,
    frame_id: int,
    T_cw: np.ndarray,
    K: np.ndarray,
    keypoints_xy: np.ndarray,
    descriptors,
) -> int:
    """
    Create and insert a keyframe into the map.
    Returns the assigned keyframe ID.
    """
    kf = Keyframe(
        kf_id=None,                 # Map assigns this
        frame_id=frame_id,
        T_cw=T_cw,
        K=K,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
        kp_to_mp=[None] * len(keypoints_xy), 
    )

    kf_id = slam_map.add_keyframe(kf)
    print(f"[MAP] Inserted KF{kf_id} (frame {frame_id})")
    return kf_id


def insert_keyframe_if_needed(
    slam_map,
    frame_id: int,
    T_cw: np.ndarray,
    K: np.ndarray,
    keypoints_xy: np.ndarray,
    descriptors,
) -> int:
    """
    Check if a keyframe for the given frame_id already exists in the map.
    If not, create and insert a new keyframe.

    Returns the keyframe ID.
    """
    existing_kf_id = get_kf_id_by_frame(slam_map, frame_id)
    if existing_kf_id is not None:
        print(f"[MAP] Keyframe for frame {frame_id} already exists as KF{existing_kf_id}")
        return existing_kf_id

    # Insert new keyframe
    kf_id = insert_keyframe(
        slam_map=slam_map,
        frame_id=frame_id,
        T_cw=T_cw,
        K=K,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
    )
    return kf_id


def initialize_map(
    slam_map,
    frame_id0: int,
    frame_id1: int,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    kp1,
    kp2,
    des1=None,
    des2=None,
):
    """
    Initialise pose-graph map with two keyframes and one odometry edge.

    Conventions:
      - Keyframe stores T_cw (world -> camera)
      - recoverPose returns motion cam0 -> cam1, so T_rel = make_T(R,t)
      - Set world frame = camera(frame_id0) at init => T_cw0 = I, T_cw1 = T_rel
    """
    T_rel = make_T(R, t)       # cam(frame_id0) -> cam(frame_id1)
    T_cw0 = np.eye(4)
    T_cw1 = T_rel @ T_cw0      # = T_rel

    kf0_id = insert_keyframe_if_needed(
        slam_map,
        frame_id=frame_id0,
        T_cw=T_cw0,
        K=K,
        keypoints_xy=kps_to_xy(kp1),
        descriptors=des1,
    )

    kf1_id = insert_keyframe_if_needed(
        slam_map,
        frame_id=frame_id1,
        T_cw=T_cw1,
        K=K,
        keypoints_xy=kps_to_xy(kp2),
        descriptors=des2,
    )

    slam_map.add_edge(
        Edge(kf_i=kf0_id, kf_j=kf1_id, T_ij=T_rel, edge_type="odometry")
    )

    print(f"[MAP] Initialised: KF{kf0_id} (frame {frame_id0}) -> KF{kf1_id} (frame {frame_id1})")
    return kf0_id, kf1_id


def edge_exists(slam_map, kf_i: int, kf_j: int, edge_type: str = "odometry") -> bool:
    """
    Check if an edge exists between two keyframes in the map.

    Parameters
    ----------
    slam_map : Map
        The SLAM map containing edges.
    kf_i : int
        ID of the first keyframe.
    kf_j : int
        ID of the second keyframe.
    edge_type : str
        Type of edge (default: "odometry").

    Returns
    -------
    bool
        True if the edge exists, False otherwise.
    """
    return any(e.kf_i == kf_i and e.kf_j == kf_j and e.edge_type == edge_type for e in slam_map.edges)


def add_map_edge(
    slam_map,
    kf_i_id: int,
    frame_j: int,
    R: np.ndarray,
    t: np.ndarray,
    kp_j,
    des_j,
    K: np.ndarray,
) -> int | None:
    """
    Adds odometry edge from existing keyframe kf_i_id to a (possibly new) keyframe at frame_j.
    Returns kf_j_id if added/exists, else None.
    """
    if kf_i_id not in slam_map.keyframes:
        print(f"[MAP][WARN] Missing KF id {kf_i_id}; cannot add odometry edge")
        return None

    T_rel = make_T(R, t)

    T_cw_i = slam_map.keyframes[kf_i_id].T_cw
    T_cw_j = T_rel @ T_cw_i

    kf_j_id = insert_keyframe_if_needed(
        slam_map=slam_map,
        frame_id=frame_j,
        T_cw=T_cw_j,
        K=K,
        keypoints_xy=kps_to_xy(kp_j),
        descriptors=des_j,
    )

    if edge_exists(slam_map, kf_i_id, kf_j_id, edge_type="odometry"):
        return kf_j_id

    slam_map.add_edge(Edge(kf_i=kf_i_id, kf_j=kf_j_id, T_ij=T_rel, edge_type="odometry"))
    print(f"[MAP] Added odometry edge KF{kf_i_id} -> KF{kf_j_id}")
    return kf_j_id


def create_mappoints_from_triangulation(
    slam_map,
    kf_i_id: int,
    kf_j_id: int,
    pts3D: np.ndarray,       # (N,3)
    idx_i: np.ndarray,       # (N,)
    idx_j: np.ndarray,       # (N,)
):
    """
    For each triangulated 3D point, create a MapPoint and attach it to:
      - kf_i at kp index idx_i[k]
      - kf_j at kp index idx_j[k]
    """
    kf_i = slam_map.keyframes[kf_i_id]
    kf_j = slam_map.keyframes[kf_j_id]

    assert pts3D.shape[0] == idx_i.shape[0] == idx_j.shape[0]

    created = 0
    for X, kp_i, kp_j in zip(pts3D, idx_i, idx_j):

        # skip if either keypoint already has a mappoint
        if kf_i.kp_to_mp[kp_i] is not None:
            continue
        if kf_j.kp_to_mp[kp_j] is not None:
            continue
            
        descriptor = kf_i.descriptors[kp_i]

        mp_id = slam_map.add_mappoint(
            xyz=X.astype(np.float64),
            descriptor=descriptor,
            observations={
                kf_i_id: int(kp_i),
                kf_j_id: int(kp_j)
            },
        )

        kf_i.kp_to_mp[kp_i] = mp_id
        kf_j.kp_to_mp[kp_j] = mp_id
        created += 1

    print(f"[MAP] Created {created} new MapPoints from triangulation")
    return created
