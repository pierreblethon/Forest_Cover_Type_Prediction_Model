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
  
<br/>
  
   <h2 align="center">Unformal Report About The Process</h3>
   
### Dataset Exploratory Analysis
The training set consists of raw unscaled data. Out of the 56 columns, one of them is an index (Id), ten of them are continuous variables and the rest consists of binary variables which have been one hot encoded. Checking the metadata, all features are integers counting no null values. Some binary variables only have one unique value (Soil_Type7 and Soil_Type15), meaning they are either constant to 0 or 1 and will therefore not help predicting the cover type. The training set is also perfectly balanced with one seventh of the total records corresponding to each cover type label. <br/><br/>
Most of the continuous features do not follow a normal distribution. Some of them are highly skewed such as Horizontal_Distance_To_Hydrology or Vertical_Distance_To_Hydrology and will require some scaling. Some features also seem highly correlated such as Slope and Aspect with the different Hillshades.<br/>
If most of the features are not highly correlated, some of them will require some feature engineering as their correlation in absolute value is close or above 0.6:
* Elevation with Horizontal_Distance_To_Roadways : 0.58
* Aspect with Hillshade_9am : -0.59
* Slope with Hillshade_Noon : -0.61
* Hillshade_Noon with Hillshade_3pm : 0.61
* Aspect with Hillshade_3pm : 0.64
* Horizontal_Distance_To_Hydrology with Vertical_Distance_To_Hydrology : 0.65
* Hillshade_9am with Hillshade_3pm : -0.78

Hillshade features and Distance features are the ones that are the most correlated with others or in between them.<br/><br/> 
Another step of this exploratory analysis was to check for outliers in the training set. Every non-categorical variable has at least one class with outliers. If these outliers might still hold relevant information, the furthest ones should be removed as they will not help generalizing the information. They might consist of exceptional trees which will not help for the prediction. In total, 2,687 outliers were found which represents 17.7% of the dataset. Not all of them will be removed in the feature engineering step as it would generate an important loss of data to train the model.<br/><br/> 
Finally, an analysis of the one-hot encoded features was done. The histogram clearly shows that some Soil_Type features are barely represented in the training set: #7 to #9, #15, #21, #25, #27, #28, #34, #36. These features might have to be removed from the model as they might not help for the prediction. The same exercise was done for the other one-hot encoded features: Wilderness_Areas. It showed how Wilderness_Area2 is underrepresented compared to the three others and might therefore not be considered when training the model.

### Data Cleaning and Feature Engineering
