# GoatAI Kitt4SME platform test

Scripts to simulate and test the entire Kitt4SME project pipeline.

### Edge system simulation

To start, we need to simulate the behavior of the edge system in the shop floor. For this purpose we can use a test
video located in `resources/sample_03.mp4`, if you want to try another video simply change the `VIDEO_PATH` setting in
the configuration file `conf/config.yaml` leaving the rest unchanged if you do not know what you are doing :).

- Once we have chosen the video and its settings we need to perform an homograph between the points of the image plan
  and the points of the floor planimetry of the shop floor, for this purpose we need nine points belonging to both
  floors. The nine points related to the shop floor are indicated in the setting `DEFAULT_REAL_WORLD_POINTS` in the
  configuration file `conf/config.yaml` (it is a 3x3 matrix of equidistant points considering the center point as the
  origin of the matrix and the others as offset from that point). The points on the image plane are chosen by the user
  when the script starts and are saved in `conf/calib.yaml`. The points on the image plane must be chosen in such a way
  to be consistent with the points defined in `DEFAULT_REAL_WORLD_POINTS`, so if in this setting we have defined nine
  points equidistant 1 meter, we will have to choose the same points on the image plane (placing markers on the floor if
  you want to facilitate the procedure).
- In addition to the points to perform the homography we need to define a polygon that represents the danger area on
  which people need to be careful to pass.

Once this first part has been defined, the system will proceed to the processing of the chosen video and will provide us
with the output of the metadata that we will send later to the context broker (whether it is on docker or on the live
platform)

### It is possible to test the functionality of the kitt4sme platform in two ways

Test the entire pipeline via the docker-compose environment, in that case check the settings in `docker-compose.yml` and run: 
```
python main.py --env docker
```
- Send the metadata directly to the previously created kitt4sme.live platform (during our tests it was instantiated
  inside a multipass virtual machine) running:
```
python main.py --env cluster
```
 Regardless of the choice, always check the IP addresses located in `kitt4sme_utils/fiware.py`



