import sys
from pathlib import Path

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from keyframe_selection.keyframe_selec import Map, Keyframe, print_map
import numpy as np

def test_add_keyframe():
    slam_map = Map()

    # Create a dummy keyframe
    kf1 = Keyframe(
        kf_id=None,
        frame_id=0,
        T_cw=np.eye(4),
        K=np.eye(3),
        keypoints_xy=np.random.rand(100, 2),
        descriptors=None
    )

    # Add the keyframe
    kf_id = slam_map.add_keyframe(kf1)

    # Check if the keyframe was added correctly
    assert kf_id == 0, "Keyframe ID should be 0"
    assert kf_id in slam_map.keyframes, "Keyframe not added to the map"
    assert slam_map.keyframes[kf_id].frame_id == 0, "Frame ID mismatch"

    # Try adding the same keyframe again (should raise an error)
    try:
        slam_map.add_keyframe(kf1)
    except ValueError as e:
        assert str(e) == "Keyframe already has kf_id=0. Set kf_id=None and let Map assign it."
    else:
        assert False, "Expected ValueError when adding duplicate keyframe"

    print("All tests passed!")

if __name__ == "__main__":
    test_add_keyframe()
    print_map(Map())
