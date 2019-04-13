# Rigid registration for 3D MRI 

Self-developed code for rigid registration during the research with [Dr.Sabuncu](https://scholar.google.com/citations?user=Pig-I4QAAAAJ&hl=en&oi=ao) and [A.Dalca](https://scholar.google.com/citations?user=zRy-zdAAAAAJ&hl=en&oi=ao). 

The code is currently in development.

## Data Augmentation
Rotate 3D image: /ext/image 

## 3D-UNet

A 3D-UNet featured with unsampling and dowsampling layer is used to learn the shift flow between two different images.

CNN outputs three 3D flows.

## Spatial Transformer

![image](https://github.com/ShouYuqing/Images/blob/master/2.png)

A 3D-spatial transformer, which is similar to the standard STN is utilized to do the registration task.

Trying to solve an optimization problem, which will get an affine transform matrix for spatial transformer to do the registration.


## Citation
Based on medical image processing library [Voxelmorph](https://arxiv.org/abs/1809.05231/) 


**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)


