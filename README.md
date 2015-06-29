# many segment :D

For the semantic segmentation service:
`roslaunch semantic_segmentation semantic_segmentation.launch`

A dummy client to test it:
`rosrun semantic_segmentation dummy_client <waypoint_name>`


#TODO:
ASAP:
* Fix flipping of normals, can we get the meta room center?
* Play around with jitter and histogram parameters.
* Try neighborhood features! (Normals, psl, color,)
* Flip and rotate the data a little for better robustness to turning of robot.

POST REV:
* Merge the two cmakelists files!
* Think about a cleaner config file handling
* Pull back other data again?

DONE?:
* Fix up label set.