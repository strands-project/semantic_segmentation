# many segment :D

For the semantic segmentation service:
`roslaunch semantic_segmentation semantic_segmentation.launch`

A dummy client to test it:
`rosrun semantic_segmentation dummy_client <waypoint_name>`


#TODO:
ASAP:
* Try larger voxels
* Improve the speed if possible? Also is this needed?

SOON:
* Flip and rotate the data a little for better robustness to turning of robot.
* Think about a cleaner config file handling
* DCRF parameters -> config file.
* Fix flipping of normals, can we get the meta room center?
* Pull back other data again?
* Fix up label set.

POST REV:
* Merge the two cmakelists files!