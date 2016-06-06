^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package semantic_segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.1 (2016-06-06)
------------------
  Fixing resource path.
* Fixing resource path.\nThis now works with rosrun a bit better.
* Update README.md
* integrate both input
* modify launch file
* Publishing for visualization.
* small changes to the feature vector.
* just intermediate backup commit
* Slightely improved voxelization parameters.
* Centering the cloud currectly for all kinds of computations.
* Changed the service to take waypoint name instead of an integrated pointcloud.
* Updated readme.
* spacing changes.
* Added code to deal with empty cloud.
* Initial version adjecency.
* Updated readme.
* Adding jitter as in vccs seeding space during training -> more data -> more stable.
* Adding histograms instead of average colors as features.
* Using the full pointcloud for computation again, not the voxelized one.
* Changing form rgb to lab
* Exposing the DCRF parameters to the config file.
* Now the voxelization is done in this service. Currently 3x3x3cm voxel centers are returned.
* Todo updates
* RF changes. Works... okayish now.
* Fixes making stuff work that didn't.
* Moved everything into this config file. Not nice, but I can't have mistakes based on two mixed up version atm.
* Exposing the other executables to the outside.
* Added some debugging output to the dummy client.
* Added DCRF smoothing.
* Several todos added to the readme.
* Fixed a big bug in the actual labeling.
  Too small voxels were not removed and the whole way indices were handles was messed up.
* Actually implemented the missing support for the vccs rectification config parameter.
  Setting this to false while training causes everything to be more similar to the integrated clouds.
  Makes the image based segmentation a lot worse and seems to slightly improve the integrated cloud labeling.
* Added an important check that would let training data go unnoticed.
  Prior to training images need to be voxelized (./voxelize --conf <conf.json>).
  If this was forgotton, CV will not complain, which results in complete images being ignored.
* Removed a confusing output from the dummy client.
* New colors for new label images.
* Space issues.
* Fixed warning.
* Added some lost code. Does nothing really like this.
* Fixed bug using the wrong amount of points.
* Normalze the frequencies.
* Actually filled the points field in the service response.
  I changed to 32bit float points as the originals are also just floats not doubles.
* Fixed some warnings.
* Removed the saving of the pointcloud causing it to crash.
* Added missing files due to overly careful .gitignore file.
* Initial commit of the new semantic segmentation.
* Emptying repo.
* Added launch file and needed code to support it.
* Now the rotation of the camera is used for the relative depth computation.
  This results better estimates for the relative depth feature of the randomized decision forests.
* Fixed the way the tilt angle is extracted and used from the tf tree.
* Added .swp files to git ignore
* Updated the config file.
  Many unneeded parts have been removed.
  These will be needed though once training a random forest is possible again.
* removed a debug output.
* Now the forest file is a parameter to the node and not set in the config.
  This is mainly done to avoid hardcoded paths in the config file.
* Added some dependencies in the package.xml
* Changed loading of calibration files.
  Previously calibrations would always be loaded from a file.
  This is not needed when loading the data from a pointcloud.
* Fixed a typo
* Initial commit of the semantic_segmentation node.
  See the README.md for initial information.
* Initial commit
* Contributors: Alexander Hermans, Karl Review, Nick Hawes, Pandoro, duanby
