# Project :bar_chart:

## **Data**

The dataset used in this project was provided by Globo with the aim of offering the most realistic and representative information possible. This data enables analysis closer to real-world scenarios, contributing to the relevance and applicability of the solutions developed in the project.

The dataset used in this project was provided by Globo in .csv format. It contains six "training" spreadsheets, where the data is organized into the following columns:

| Column                    | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `userId`                  | User ID                                              |
| `userType`                | Logged-in or anonymous user                          |
| `historySize`             | Number of news articles read by the user             |  
| `history`                 | List of news articles visited by the user            |  
| `timestampHistory`        | Timestamp when the user visited the page             |  
| `numberOfClicksHistory`   | Number of clicks on the article                      |  
| `timeOnPageHistory`       | Time (in ms) the user spent on the page              |  
| `scrollPercentageHistory` | Percentage of the article viewed by the user         |  
| `pageVisitsCountHistory`  | Number of times the user visited the article         |  
| `timestampHistory_new`    | Timestamp when the user visited the page (new)        |  

???+ info

    The timestampHistory_new column was not described by Globo. Therefore, the difference in relation to the "timestampHistory" column is not clearly defined.


Another file provided by Globo consisted of 3 .csv files named "items". These files contained the following information:

| Column      | Description                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `page`      | News article ID. This is the same ID that appears in the "history" column of the training .csv files explained earlier |
| `url`       | News article URL                                                                                                       |
| `issued`    | Date when the news article was created                                                                                 |  
| `modified`  | Last date when the news article was modified                                                                           |  
| `title`     | News article title                                                                                                     |  
| `body`      | News article body                                                                                                      |  
| `caption`   | News article subtitle                                                                                                  |  

The data presented are the ones initially provided. However, it is possible to extract additional information through treatments applied to these data. The treatments performed and the potential new data generated from them will be explained later, should they be used.

## **Model Training**: 

## **Model Saving**: 

## **API for Predictions**: 

## **Packaging with Docker**: 

## **API Testing and Validation**:

## **Deployment **: 
