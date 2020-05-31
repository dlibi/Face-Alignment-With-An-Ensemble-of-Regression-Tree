# Face-Alignment-With-An-Ensemble-of-Regression-Tree

Face alignment is a process that estimates the face’s landmarks position or facial shape from face images, and it is widely used in research and commercial applications, including, object pose estimation, face recognition, 3D reconstruction and automatic face beautification.

  This is a project work base on:
  
  (1) Kazemi V, Sullivan J One millisecond face alignment with an ensemble of regression tree. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2014.
  
  (2) menpo package
  
  (3) other former code works.
  
  The theory that we use here is the ensemble of regression model from the paper of (1). And the code is produced strictly on it. However, there is still problems like parameters need to figure out. Dataset that used here is IBUG 300M which is classified into indoor and outdoor. 50 out of 300 from each class is used as test.
  
  The final error rate is controled below 0.1 in the test dataset. Some of the results are stored in the Models folder as well as the estimated model.
 
It is too big to upload the images, and the data that I use can be reached from:
link:https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
