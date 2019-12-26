# Disaster Response Pipeline Project
___________________________________________________________________
### A Udacity Data Scientist Nanodegree Project



### Table of Contents

1.[Project_Description](#Project_Description)

2.[Installation](#Installation)

3.[Executing_Program](#Executing_Program)

4.[Authors](#Authors)

5.[LICENSE](#LICENSE)

6.[Acknowledgements](#Acknowledgements)

7.[Screenshots](#Screenshots)


<a name="Project_Description"></a>

## Project Description

This project is a part of the Udacity Data Scientist Nanodegree. This project is conating dataeset from pre-labeled tweets and messages from real-life disasters. The aim of this project is
to build a model that categorizes messages.

<a name="Installation"></a>


## Installation

python3 should be installed including these python liberals:
NumPy
SciPy
Pandas
Sciki-Learn
Flask
Plotly
NLTK
SQLalchemy

<a name="Executing_Program"></a>


## Executing Program
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="Authors"></a>


### Authors 


##### Authors * [Abdulelah Asiri](https://github.com/Abdol9900)


<a name="LICENSE"></a>

### LICENSE 

Abduleha Asiri,

License under the [MIT License](LICENSE)




<a name="Acknowledgements"></a>

### Acknowledgements 

 Must give credit to Figure Eight for providing messages dataset to train the model, Also Udacity for providing such a useful data scientist course.


<a name="Screenshots"></a>

### Screenshots


### 1-The main page shows two graphs that represnet training dataset.

![pic1](https://github.com/Abdol9900/Disaster-Response-Pipeline-Project/blob/master/pic1.JPG)

### 2. Here you can type a sentence to test the ML model performance

![pic2](https://github.com/Abdol9900/Disaster-Response-Pipeline-Project/blob/master/pic3.JPG)

### 3. clicking Classify message to see the categories to show where the sentence related to.

![pic3](https://github.com/Abdol9900/Disaster-Response-Pipeline-Project/blob/master/pic5.JPG)
