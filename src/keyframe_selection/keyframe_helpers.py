import numpy as np
from .keyframe_selec import Keyframe, Edge, MapPoint
import cv2 as cv
from triangulation.triangulation import triangulate_from_data


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
    des1,
    des2,
):
    """
    Initialise pose-graph map with two keyframes and one odometry edge.

    Conventions:
      - Keyframe stores T_cw (world -> camera)
      - recoverPose returns motion cam0 -> cam1, so T_rel = make_T(R,t)
      - Set world frame = camera(frame_id0) at init => T_cw0 = I, T_cw1 = T_rel
    """
    T_rel = make_T(R, t)       # cam(frame_id0) -> cam(frame_id1)
    T_cw0 = np.eye(4, dtype=np.float64)  # cam(frame_id0) -> world
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

    # Ensure the number of 3D points matches the number of indices
    assert pts3D.shape[0] <= idx_i.shape[0] and pts3D.shape[0] <= idx_j.shape[0], \
        "[MAP] Mismatch between triangulated points and keypoint indices"

    # Filter idx_i and idx_j to match the filtered 3D points
    idx_i = idx_i[:pts3D.shape[0]]
    idx_j = idx_j[:pts3D.shape[0]]

    created = 0
    for X, kp_i, kp_j in zip(pts3D, idx_i, idx_j):

        # Skip if either keypoint already has a mappoint
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


def run_pnp_for_frame(slam_map, keypoints, descriptors, K):

    if len(slam_map.mappoints) == 0:
        return None, 0

    # Build descriptor array from map
    mp_ids = []
    mp_des = []
    mp_xyz = []

    for mp_id, mp in slam_map.mappoints.items():
        mp_ids.append(mp_id)
        mp_des.append(mp.descriptor)
        mp_xyz.append(mp.xyz)

    mp_des = np.array(mp_des, dtype=np.uint8)
    mp_xyz = np.array(mp_xyz, dtype=np.float64)

    # BF matcher for ORB
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(mp_des, descriptors)

    if len(matches) < 6:
        return None, 0

    object_points = []
    image_points = []

    for m in matches:
        object_points.append(mp_xyz[m.queryIdx])
        image_points.append(keypoints[m.trainIdx].pt)

    object_points = np.array(object_points)
    image_points = np.array(image_points)

    success, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        image_points,
        K,
        None,
        reprojectionError=3.0,
        confidence=0.99,
        iterationsCount=100
    )

    if not success or inliers is None or len(inliers) < 10:
        return None, 0

    R, _ = cv.Rodrigues(rvec)
    T = make_T(R, tvec.flatten())

    return T, len(inliers)


def run_pnp_for_frame2(slam_map, keypoints, descriptors, K,
                      min_matches=20, min_inliers=15, ratio=0.75):
    """
    Returns:
      Tcw (4,4) or None
      num_inliers (int)
      inlier_kp_to_mp (dict): {kp_idx_in_current_frame: mp_id}
    """

    if descriptors is None or len(keypoints) == 0:
        return None, 0, {}

    if len(slam_map.mappoints) == 0:
        return None, 0, {}

    # Build descriptor/xyz arrays from map
    mp_ids = []
    mp_des = []
    mp_xyz = []
    for mp_id, mp in slam_map.mappoints.items():
        if mp.descriptor is None:
            continue
        mp_ids.append(mp_id)
        mp_des.append(mp.descriptor)
        mp_xyz.append(mp.xyz)

    if len(mp_ids) < 6:
        return None, 0, {}

    mp_des = np.asarray(mp_des, dtype=np.uint8)           # (M, D)
    mp_xyz = np.asarray(mp_xyz, dtype=np.float64)         # (M, 3)

    # KNN match MapPoint descriptors -> current frame descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(mp_des, descriptors, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None, 0, {}

    # Build correspondences, enforcing unique 2D keypoints
    used_kp = set()
    object_points = []
    image_points = []
    kp_to_mp = {}  # kp_idx -> mp_id

    for m in sorted(good, key=lambda x: x.distance):
        kp_idx = m.trainIdx      # index into current keypoints/descriptors
        mp_row = m.queryIdx      # index into mp_xyz/mp_ids
        if kp_idx in used_kp:
            continue
        used_kp.add(kp_idx)

        object_points.append(mp_xyz[mp_row])
        image_points.append(keypoints[kp_idx].pt)
        kp_to_mp[kp_idx] = mp_ids[mp_row]

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points  = np.asarray(image_points,  dtype=np.float64).reshape(-1, 2)

    if object_points.shape[0] < 6:
        return None, 0, {}

    # Solve PnP
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        image_points,
        K,
        None,
        iterationsCount=200,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv.SOLVEPNP_EPNP
    )

    if (not ok) or inliers is None or len(inliers) < min_inliers:
        return None, 0, {}

    # Optional refine on inliers
    inl = inliers[:, 0]
    rvec, tvec = cv.solvePnPRefineLM(
        object_points[inl],
        image_points[inl],
        K, None,
        rvec, tvec
    )

    R, _ = cv.Rodrigues(rvec)
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = R
    Tcw[:3, 3] = tvec.reshape(3)

    # Keep only kp_to_mp entries that were inliers
    # We need the mapping from inlier row -> kp_idx, so rebuild:
    # (simple way: store kp_idx list aligned with object_points)
    # Let's do it properly:

    # Rebuild aligned kp_idx list
    kp_idx_list = list(kp_to_mp.keys())
    inlier_kp_to_mp = {kp_idx_list[r]: kp_to_mp[kp_idx_list[r]] for r in inl}

    return Tcw, int(len(inliers)), inlier_kp_to_mp