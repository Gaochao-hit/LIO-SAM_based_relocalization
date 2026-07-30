## Abstract
A simple version of system that can relocalize in a built map is developed in this repository. The sysytem is bsed on LIO_SAM.
The repository is developed based on the origional version of LIO-SAM in which the GPS is not fused.

## Run the package
1. Make sure the map should be saved in the right folder:
```
Firstly, you need to run LIO-SAM, and then save the map in the default folder
```

2. Run the launch file:
```
roslaunch lio_sam run_relocalize.launch
```

3. Play existing bag files:
```
rosbag play your-bag.bag
```

 A video of the demonstration of the method can be found on [YouTube](https://youtu.be/PRsH8SpuSIc)
 ## Notes

  - **Initialization:** During the initialization stage, had better keep the robot still. Or if you play bags, fistly play the bag for about 0.5s, and then pause the bag until the initialization succeed. The initialization method requres you to give it initial guesses by hand on the the Rviz.

## Acknowledgement
Thanks to the authors of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) for code release.
