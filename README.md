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
Regarding data cleaning, we start by removing some irrelevant features as we found that there were two types of soils with only zeros (Soil_Type7 and Soil_Type15). Once we removed those features, we checked for outliers within each class defining the interquartile range per class. Our definition of and outlier for this project will be any value over two times the third quartile of the selected class, or below two times the first quartile of the selected class. With this definition, we can count the number of outliers per class. The percentage of outliers is around 6% of the data, so it was decided to remove them as there is enough data to train our algorithms. After removing outliers, the balance of the cleaned dataset was rechecked and still appeared to be well balanced.<br/><br/> 
Now it was time for feature engineering. The first thing that was realized, as explained in the exploratory analysis, is that the two variables referring to hydrology (Vertical and Horizontal_Distance_To_Hydrology) were highly correlated. It was thought that if we adjust the hydrology point in one of the edges of a right triangle, we can infer the hypotenuse with the vertical and the horizontal distance, the hypotenuse being the “real” distance to water. This relation can be calculated as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?real\;distance&space;=&space;\sqrt{vertical\;&space;distance^{2}&plus;horizontal\;&space;distance^{2}}">

The new feature Hypotenuse_Distance_To_Hydrology allows us to find the relation between two of the original features which can now be dropped from the training set.<br/>
However, our main effort was to clarify and find a relation between the different hill-shades features. The initial findings uncovered what was the hillshade index in real life: a topographical index to develop and produce a grayscale 3D representation of a terrain surface, with the sun's relative position taken into account to shade the image. The index ranges from 0 to 255, 0 being a surface with a low shade index and 255 a surface with a high shade index (Ref. 1).<br/>
In other words, 0 is telling that the surface is sun-faced, and 255 that the surface is not sun faced. The intermediate values show a displacement from more to less sun incidence. The next figure shows what the hillshade index does when it is used with a topographical software:

<p align="center">
  <a href="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-0A189843-6ADF-4FBF-A3A4-A51F515B404A-web.png">
    <img src="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-0A189843-6ADF-4FBF-A3A4-A51F515B404A-web.png" width="300">
  </a>
<br />
  
However, there are two other features that together describe that same sun-facing property: Slope and Aspect. In a 3D structure such as a real forest, slope can be seen as the gradient of change between the points that define two different lines. The following image show the slope of a 3D object:
  
  <p align="center">
  <a href="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-0A189843-6ADF-4FBF-A3A4-A51F515B404A-web.png">
    <img src="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-0A189843-6ADF-4FBF-A3A4-A51F515B404A-web.png" width="300">
  </a>
<br />
 
In this example, A, B, Q and P define a plane, and the change between PQ and AB define the slope of the plane.<br/> 
The aspect is the direction of the plane with respect to some arbitrary zero, and in this case, since the aspect is defined in azimuth degrees, in respect to the north.<br/> <br/> 
To summarize, with the slope (Slope) and the direction of the slope relative to the north (Aspect), by combining them together, we can create a new feature that gathers the same information as the hillshade index. This situation shows that, in the training set, five features are gathering similar information (Slope, Aspect, Hillshade_9am, Hillshade_Noon and Hillshade_3pm). Therefore, we have different options: use the hillshade index by combining the three given in the training set, or use a combination of Slope and Aspect. Incoming radiation for a given point at a given time is either a function of the slope and aspect of that point, or the hillshade index. However, it is not a combination of the two indexes together (Ref. 2)<br/> <br/> 
One could now ask: why is all this information important for our model? The answer is: because, as we can guess for the forest studied, south facing slopes are sunnier and this higher incoming radiation is strongly correlated with the type of tree growing in each area.<br/> 
After reading several papers and according to some studies, it appears to be better to use hillshade to describe sun exposition rather than a combination of slope and aspect (Ref. 3).<br/> 
With all that information in mind, the only point to clarify now is how to connect the three hill-shades together, knowing it is the measure of the same terrain from three different points of view. In this case, we believe that to improve the model’s prediction, converting the hillshade index into an illumination coefficient (a new Sun_Illumination feature) would be a good idea. This coefficient would take into account the sun incidence of each tree, and then compare all the trees of the forest in between them.<br/> <br/> 
The first step is to convert the hillshade index into a homemade index considering sun exposition rather than shade (Hillsun):
