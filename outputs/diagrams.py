from IPython.display import Markdown, display

mermaid_diagram = r"""
```mermaid
flowchart TD

    A[Start / Input RGB Image Sequence] --> B[Stage 1: Feature Detection & Matching<br>• ORB keypoints<br>• FLANN + ratio test<br>• Histogram & LK filtering]

    B --> C[Stage 2: Pose Estimation<br>• Compute Essential Matrix<br>• RANSAC removes outliers<br>• recoverPose → R, t<br>• Returns inlier mask]

    C --> D[Stage 3: Triangulation<br>• Use R, t + inlier matches<br>• cv.triangulatePoints<br>• Filter invalid 3D points<br>→ Local 3D point cloud]

    D --> E[Keyframe Selection<br>• Enough inliers?<br>• Good inlier ratio?<br>• Low reprojection error?<br>→ Accept / Reject frame]

    E --> F[Stage 4: Point Cloud Alignment<br>• Chain camera poses<br>• Transform each local cloud<br>• (Optional) ICP refinement<br>→ Global 3D point cloud]

    F --> G[Stage 5: Visualisation<br>• Render global point cloud<br>• Show camera trajectory]

    G --> H[Output<br>• 3D Map<br>• Camera Path]
"""


⚠ NOTE  
This does NOT display in Jupyter by default.  
You need a renderer (like `mermaid-js` plugin or Quarto/MkDocs).

---

# ✅ **3. HTML version (for reports, websites, MkDocs)**

```html
<div class="mermaid">
flowchart TD

    A[Start / Input RGB Image Sequence] --> B[Stage 1: Feature Detection & Matching<br>• ORB keypoints<br>• FLANN + ratio test<br>• Histogram & LK filtering]

    B --> C[Stage 2: Pose Estimation<br>• Compute Essential Matrix<br>• RANSAC removes outliers<br>• recoverPose → R, t<br>• Returns inlier mask]

    C --> D[Stage 3: Triangulation<br>• Use R, t + inlier matches<br>• cv.triangulatePoints<br>• Filter invalid 3D points<br>→ Local 3D point cloud]

    D --> E[Keyframe Selection<br>• Enough inliers?<br>• Good inlier ratio?<br>• Low reprojection error?<br>→ Accept / Reject frame]

    E --> F[Stage 4: Point Cloud Alignment<br>• Chain camera poses<br>• Transform each local cloud<br>• (Optional) ICP refinement<br>→ Global 3D point cloud]

    F --> G[Stage 5: Visualisation<br>• Render global point cloud<br>• Show camera trajectory]

    G --> H[Output<br>• 3D Map<br>• Camera Path]
</div>
