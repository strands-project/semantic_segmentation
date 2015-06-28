# many segment :D

For the semantic segmentation service:
`roslaunch semantic_segmentation semantic_segmentation.launch`

A dummy client to test it:
`rosrun semantic_segmentation dummy_client <waypoint_name>`


#TODO:
* Flip and rotate the data a little for better robustness to turning of robot.
* Merge the two cmakelists files! SERIOUSLY!
* Think about a cleaner config file handling
* DCRF parameters -> config file.
* Voxel sizes.
* Fix flipping of normals, can we get the meta room center?
* Pull back other data again?
* Improve the speed if possible? Also is this needed?
* Fix up label set.