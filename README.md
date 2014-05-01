semantic_segmentation
=====================
Run with:

```bash
roslaunch semantic_segmentation segment_metric_maps.launch
```

There are several parameters which are set to default values in the launch file.

- `input_topic` The input topic, should always be an intermediate map from the local metric maps node. 
- `ptu_tilt_motor_frame` The tf-frame for the ptu tilt motor.
- `ptu_hinge_frame` The tf-frame for the ptu hinge. Although the optical frame for the head xtion should be fine as well.
- `semantic_config_file` Path to the config file. 
- `semantic_forest_model` Path to the randomized decision forest file. 
