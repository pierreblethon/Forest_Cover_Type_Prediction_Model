  <h1 align="center">Forest Cover Type Prediction Model</h3>

<p align="center">
  <a href="https://cpw.state.co.us/placestogo/parks/ParkPhotos/StateForest08.jpg">
    <img src="https://cpw.state.co.us/placestogo/parks/ParkPhotos/StateForest08.jpg" width="600">
  </a>
<br />
  
  ### About The Project

* This project presents an attempt to predict the kind of tree cover from four wilderness areas located in the Roosevelt National Forest of northern Colorado.
* This project started with the motivation to create my own machine learning pipeline and get a sense of what kaggle competitions consist of.
* The dataset consists of two files:
   * A labeled training set with 15 120 records
   * An unlabeled test set with 565 892 records
  
### Main Results & Findings
  
* Random Forest is the algorithm giving the best prediction according to cross-validation.
  * Accuracy is 85.3%
  * Recall is 85.0%
  * Precision is 84.7%
* The majority of records seems to be tree covers of type 1 and type 2.
  * These cover types are the ones our model struggles predicting the most according to confusion matrixes. 
