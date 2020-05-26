# Panorama-Pseudo-LiDAR
We presented a system to generate the panorama pseudo-LiDAR point, with multiple monocular photos taken from different orientations serving as input. In this system, we successfully recovered and sparsified the point cloud from each direction and aligned them with the transformation matrix calculated from image features.

The project can be decomposed to serval parts. 

Folder **dataset** contains the method to load data from KITTI, Waymo Open dataset and custom dataset.

Folder **src** contains the source python files for this project. In these files, we wrote API for liteflow, PointRCNN, Monodepth2 and Realsense camera.
### Framework
The whole pipeline is shown in the figure.
<img src="imgs/report_pipeline.jpg" height="280" alt="Complexity" align=center>

### Test on KITTI and Waymo

```bash
python main.py
```
### Collect data from Realsense camera

```bash
python bino_api.py
```

### Visualize results

```bash
python open3d_utils.py
```

### Results

#### KITTI

Source images:

<img src="imgs/kitti_image.png" height="120" alt="Complexity" align=center>

Point cloud recovered from setero images and comparison with ground truth LiDAR points.

<img src="imgs/stereo_recover.png" height="180" alt="Complexity" align=center>

Detection results

<img src="imgs/stereo_kitti_detection.png" height="180" alt="Complexity" align=center>

#### Waymo

Source images:

<img src="imgs/waymo_data.png" height="80" alt="Complexity" align=center>

Panorama point clouds recovered from five images with monodepth2 as depth prediction model.

<img src="imgs/waymo_result.png" height="160" alt="Complexity" align=center>


#### Custom data

Source images:

<img src="imgs/mono_data.png" height="120" alt="Complexity" align=center>

Fusion result:

<img src="imgs/mono_pano.png" height="180" alt="Complexity" align=center>

ICP fusion results details:

<img src="imgs/details.png" height="140" alt="Complexity" align=center>