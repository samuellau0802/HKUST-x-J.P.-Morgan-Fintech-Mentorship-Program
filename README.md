# HKUST-x-J.P.-Morgan-Fintech-Mentorship-Program

## **Scraper**
*stocktwits_scraper.py* : a scraper to scrap stocktwits tweets by inputting stock symbol and earliest date

*yahoo_scraper.py* : a scraper to scrap yahoo finance conversation

*check_scrapped.ipynb* : check if all the stocks are being scraped

## **Data Exploration**
*stocktwits_explore.ipynb* : plotting graphs, tf-idf, wordclouds, frequencies

*data_verifying.ipynb* : check multiple sources of data

## **Data Cleaning**
*stocktwits_cleaning.ipynb* : cleaning of the scraped data (remove symbols, tokenize etc.)

### **Data Cleaning: Snorkel**
*data_labelling.ipynb* : remove useless data

*tfidf_function.ipynb* : find tfidf words and remove in snorkel

## **Model Testing: Classification**
*model_training.ipynb* : model training and testing, hyperparameters testing

*model_testing.ipynb* : testing different pipelines

**.py* : different ML models

## **Sentiment**
*to_dataset.ipynb* : from our dataset to the required format
*fine_tune_sentiment.ipynb* : transfer learning of sentiment