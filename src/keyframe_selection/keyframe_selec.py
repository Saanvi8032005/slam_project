from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

#   T_ij = T_cw_j @ np.linalg.inv(T_cw_i)
#   maps points from camera i to camera j


@dataclass
class Keyframe:
    """
    Represents a keyframe in the SLAM system.

    Attributes:
        kf_id (int): Unique ID for the keyframe.
        frame_id (int): Frame ID from the input data.
        T_cw (np.ndarray): Transformation matrix (4x4) from world to camera coordinates.
        K (np.ndarray): Camera intrinsic matrix (3x3).
        keypoints_xy (np.ndarray): 2D coordinates of keypoints in the image (N, 2).
        descriptors (Optional[np.ndarray]): Feature descriptors for the keypoints (N, D).
        is_loop_candidate (bool): Indicates if the keyframe is a loop closure candidate.
    """
    kf_id: Optional[int]  # Assigned when added to the map
    frame_id: int
    T_cw: np.ndarray          # (4,4)
    K: np.ndarray             # (3,3)
    keypoints_xy: np.ndarray  # (N,2)
    descriptors: Optional[np.ndarray]  # (N,D) (can be None for now)
    is_loop_candidate: bool = False


@dataclass
class Edge:
    """
    Represents an edge between two keyframes in the SLAM graph.

    Attributes:
        kf_i (int): ID of the first keyframe.
        kf_j (int): ID of the second keyframe.
        T_ij (np.ndarray): Measured relative transformation (SE3 or Sim3).
        weight (float): Weight or information scalar for the edge.
        edge_type (str): Type of edge, e.g., "odometry" or "loop".
    """
    kf_i: int
    kf_j: int
    T_ij: np.ndarray      # measured relative transform (SE3 or Sim3)
    weight: float = 1.0   # or information scalar
    edge_type: str = "odometry"  # or "loop"


class Map:
    """
    Represents the map in the SLAM system, managing keyframes and edges.

    Attributes:
        keyframes (Dict[int, Keyframe]): Dictionary of keyframes indexed by their IDs.
        edges (List[Edge]): List of edges representing relationships between keyframes.
        _next_kf_id (int): Internal counter for generating unique keyframe IDs.
    """
    def __init__(self):
        self.keyframes: Dict[int, Keyframe] = {}
        self.edges: List[Edge] = []
        self._next_kf_id = 0

    def add_keyframe(self, kf: Keyframe) -> int:
        """
        Adds a new keyframe to the map.

        Args:
            kf (Keyframe): The keyframe to add.

        Returns:
            int: The ID of the added keyframe.

        Raises:
            ValueError: If a keyframe with the same ID already exists.
        """
        if kf.kf_id is not None:
            raise ValueError(
                f"Keyframe already has kf_id={kf.kf_id}. "
                "Set kf_id=None and let Map assign it."
            )

        new_id = self._next_kf_id
        self._next_kf_id += 1

        kf.kf_id = new_id
        self.keyframes[new_id] = kf
        return new_id

    def add_edge(self, edge: Edge) -> None:
        """
        Adds a new edge between two keyframes.

        Args:
            edge (Edge): The edge to add.

        Raises:
            ValueError: If the keyframes referenced by the edge do not exist.
        """
        if edge.kf_i not in self.keyframes or edge.kf_j not in self.keyframes:
            raise ValueError(f"Cannot add edge: Keyframes {edge.kf_i} or {edge.kf_j} do not exist in the map.")
        self.edges.append(edge)

    def remove_keyframe(self, kf_id: int) -> None:
        if kf_id not in self.keyframes:
            raise KeyError(f"Keyframe {kf_id} does not exist")

        # Remove edges connected to this keyframe
        self.edges = [
            e for e in self.edges
            if e.kf_i != kf_id and e.kf_j != kf_id
        ]

        # Remove keyframe
        del self.keyframes[kf_id]


def print_map(slam_map):
    print("\n================ MAP STATE ================")

    print(f"Num keyframes: {len(slam_map.keyframes)}")
    print(f"Num edges:     {len(slam_map.edges)}")

    if False:
        print("\nKeyframes:")
        for kf_id, kf in slam_map.keyframes.items():
            print(f"  KF{kf_id}: frame_id={kf.frame_id}")
            print(f"    T_cw:\n{kf.T_cw}")

        print("\nEdges:")
        for e in slam_map.edges:
            print(
                f"  Edge {e.kf_i} -> {e.kf_j} "
                f"type={e.edge_type} weight={e.weight}"
            )
            print(f"    T_ij:\n{e.T_ij}")

    print("===========================================\n")
