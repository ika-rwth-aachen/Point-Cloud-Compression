# SNNRMSE Metric Node

A metric to evaluate the "distance" between two point clouds. 
Definition of the metric can be found in the paper [Real-time Point Cloud Compression](https://cg.cs.uni-bonn.de/aigaion2root/attachments/GollaIROS2015_authorsversion.pdf).

![Equation](https://latex.codecogs.com/gif.latex?\text{RMSE}_{\text{NN}}(P,Q)=\sqrt{\sum_{p\in{P}}(p-q)^2/\lvert%20P%20\rvert})

![Equation](https://latex.codecogs.com/gif.latex?\text{SNNRMSE}(P,Q)=\sqrt{0.5*\text{RMSE}_\text{NN}(P,Q)+0.5*\text{RMSE}_\text{NN}(Q,P)})

## Usage
The node subscribes to topic `/points2` for the original point cloud, and the topic `/decompressed` for the decompressed 
point cloud. Messages should be of type `PointCloud2`. Only two point clouds pairs with identical stamps
are evaluated.
