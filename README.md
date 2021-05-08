# Disaster Response Pipeline Project

The goal of this project is to develop an ETL and machine learning pipeline to produce a web app which can evaluate messages relating to disasters and provide the correct categorization of the situation and possible response needed.

The project is made up of 3 parts, one portion to load and clean the initial data, a machine learning pipeline to build, train, evaluate, and save a model, and a Flask web app to deploy the model and provide message category results to disaster relief agencies.

## Files:
	./app/
	* 'run.py' - Code for running Flask web app
	* templates - Folder containing template information for web app
	
	./data/
	* 'disaster_messages.csv' - Original dataset of message text
	* 'disaster_categories.csv' - Original dataset of message categories
	* 'DisasterResponse.db' - SQL database of cleaned messages output from process_data.py
	* 'process_data.py' - Code for loading, cleaning, and outputting message and category data to SQL database
	
	./models/
	* 'train_classifier.py' - Code for building, training, and evaluating model

## Data:
	The data used in this project was collected from [Figure Eight](https://www.figure-eight.com/) and consists of messages sent during disaster events and their respective category information. There were 36 total categories included.
	
## Model:
	The resulting model is a multi-output classifier, outputting results of a given message for each of the 36 categories. It uses a random forest algorithm and achieved an overall Hamming Loss of 0.054.
	
## Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
