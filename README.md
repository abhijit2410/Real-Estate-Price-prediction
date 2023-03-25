# Real-Estate-Price-prediction
<p>
    <img height="400px" width="900px" src="https://user-images.githubusercontent.com/77913276/227735067-68ea640b-a036-4187-b96e-53810cc0f46d.jpg">
</p>
This repository contains a real estate price prediction model that uses machine learning algorithms to predict the sale price of real estate properties at Boston. The model is based on a dataset that includes various features of properties, such as the **number of bedrooms, bathrooms, square footage, and location**.

The project is built using Python programming language and utilizes several libraries, including **Pandas, Numpy, Scikit-learn, and Matplotlib**. The dataset used in this project is publicly available and has been preprocessed to ensure its accuracy and completeness.

## Installation

To run the model, you will need to install the following libraries:

* Pandas
* Numpy
* Scikit-learn
* Matplotlib

## Data Source and Data Featues
The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston.

> # The features can be summarized as follows:

* CRIM: This is the per capita crime rate by town
* ZN: This is the proportion of residential land zoned for lots larger than 25,000 sq.ft.
* INDUS: This is the proportion of non-retail business acres per town.
* CHAS: This is the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
* NOX: This is the nitric oxides concentration (parts per 10 million)
* RM: This is the average number of rooms per dwelling
* AGE: This is the proportion of owner-occupied units built prior to 1940
* DIS: This is the weighted distances to five Boston employment centers
* RAD: This is the index of accessibility to radial highways
* TAX: This is the full-value property-tax rate per $10,000
* PTRATIO: This is the pupil-teacher ratio by town
* B: This is calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town
* LSTAT: This is the percentage lower status of the population
* MEDV: This is the median value of owner-occupied homes in $1000s

> # An overview of the original dataset
![data demo](https://user-images.githubusercontent.com/77913276/227734707-1bdc1ba8-0aaf-4145-9ce2-6fd6945fb471.jpg)

> # Exploratory Data Analysis
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    sns.pairplot(data, size=2.5)
    plt.tight_layout()
    
  <p>
    <img height="500px" width="700px" src="https://user-images.githubusercontent.com/77913276/227734764-35e56f28-dd52-403e-b651-11523d8d7717.jpg">
  </p>

> # Correlation Matrix
    cm = np.corrcoef(data.values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
![corelation](https://user-images.githubusercontent.com/77913276/227734953-abcae02b-dd91-4268-b7a8-d3d619e118fb.jpg)


## Acknowldegements

* The dataset used in this project is publicly available and can be found on Kaggle.
* The code for this project was inspired by the work of several machine learning researchers and practitioners.
