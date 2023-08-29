# ML-A1-JiewenShen
This is a website test based on pycharm's forecast price
 

1. I used linear regression and LGBM (Light Gradient Boosting Machine) to predict car prices.
2. Thinking steps:

Import the data from the training set and the test set separately, and then merge them.

-- Analyze the combined data set (it can be concluded that the new_price of the car is missing in a large area, and the missing value of other features is relatively small).

Process the merged data: remove duplicate values and remove the brand column.

- Dealing with missing data:

(1) For seat, the value cannot be 0, so it is deleted;

(2) For power,engine and mileage, the median is used to fill in the missing data;

- Select different parameters and prices for classification comparison, such as: year and price, region and price, mileage and price, fuel type and price, engine displacement and price, engine power and price, number of seats and price.

-- Divide the data.

-- Build a model:

(1) Use linear regression

(2) Using LGBM to predict prices.

 
      
      
   After analysis, we understand that the car brand and the price have no direct impact, the brand effect will make the car price there is a serious phenomenon. Fuel type, year, region, mileage, fuel type, engine displacement, engine power, number of seats, etc. have a direct relationship with the price. In the aspect of prediction analysis, in addition to classification model, I have learned about the nearest neighbor method (KNN algorithm), in order to determine the "nearest neighbor", it is necessary to define the distance function; Incremental learning is automatic as more data arrives (old data can also be deleted). But its weakness is that it doesn't handle a large number of dimensions very well.
