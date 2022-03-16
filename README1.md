# LINEAR REGRESSION : SHARED BIKE DEMAND ASSIGNMENT
> A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19.They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends.The company wants to know:

> 1. Which variables are significant in predicting the demand for shared bikes.
> 2. How well those variables describe the bike demands 

# Business Goal:
> You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary  with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 


## Table of Contents
* [Overview and Problem Statement](#general-information)
* [Python, Pandas, Numpy, Matplotlib,scikit-learn, statsmodels, Exploratory Data Analysis (EDA) techniques, and Seaborn for Visulaisation](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Programming assignment wherein you have to build a multiple linear regression model for the prediction of demand for shared bikes.
- Shared bike renting company Boombikes has suffered considerable dip in their revenues due to Corona pandemic. They contracted a consulting company to understand the factors leading to this revenue dip and provide them with direction to bounce back on their revenues.
- Business goal is to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features
- Dataset used is 'day.csv' wherein we have variables season, yr, mnth, weekday, working day, weathersit, temp, hum, windspeed and cnt. The model is built taking 'cnt' as the target variable.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Demand for bikes dependant on variables yr, holiday, spring, light_rain_snow_thunderstrm, mist_cloud, months (3,5,6,8,9,7,10) and Sun
- Demand increases in months 3,5,6,8,9,7,10 and by yr. 2019 saw higher demand than compared to 2018.
- Demand decreases on holiday, spring, sunday, mist_cloud and light_rainsnow_thunderstorm

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies
- Jupyter Notebook
- Github
- Numpy, Pandas, Matplotlib, Scikit-learn, statsmodels, Seaborn

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements/References
- Upgrad Course Materials
- StackOverflow
- Towardsdatascience.com


## Contact
Created by [@githubusername] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
