# Code Citations

## License: unknown
https://github.com/maxkferg/object-reconstruction/tree/37321936588a839d3b533830262ba150a0df0081/calibrate.py

```
w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
```

