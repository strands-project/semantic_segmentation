# many segment :D

For the semantic segmentation service:
`roslaunch semantic_segmentation semantic_segmentation.launch`

A dummy client to test it:
`rosrun semantic_segmentation dummy_client <waypoint_name>`


#TODO:
ASAP:
* Try larger voxels
* Improve the speed if possible? Also is this needed?
* * Try neighborhood features! (Normals, psl, color,) 

SOON:
* DCRF parameters -> config file.
* Flip and rotate the data a little for better robustness to turning of robot. Also jitter segmentation parameters.
* Fix flipping of normals, can we get the meta room center?

POST REV:
* Merge the two cmakelists files!
* Think about a cleaner config file handling
* Pull back other data again?

DONE?:
* Fix up label set.