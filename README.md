semantic_segmentation
=====================
Currently the node still needs some parameters. 

Run with:

'''
rosrun semantic_segmentation semantic_segmentation <pointcloud topic> <tf-frame for camera> <config_file>
'''

The config file still needs some adaption in order to point to the correct rdf.dat file for the randomized decision tree data. Both the config file and the rdf file are located in the data folder. 

The node will currently open an opencv window to show the result. This will be adapted to whatever needs to happen. 

