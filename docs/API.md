# API Documentation :pushpin:

## **API**: 

The API created is aimed to return recommended content to an specific user whether it exists or not already in our database.
This API contains one endpoint **recommend** and it expects a json with a userID as you can see from the example below.
```json
{
  "user": "0004e1ddec9a5d67faa56bb734d733628a7841c10c7255c0c507b7d1d4114f06"
}
``` 
Once a user is provided the API can return 2 possible outcomes:

1. The user already exists and has some history, in this case we will provide the recommendations based on the user last interactions

2. The user does not exists meaning this is a fresh new access and we have no history to base our recommendations, then we will recommend a rank with the 15 most accessed news.

## **Packaging with Docker**: 
To serve the API we are using docker, and this is achieve through bentoml and its *bentofile.yml*.
The *bentofile.ylm* is used to build a docker image containing hour API and the inner logic for, later the API is deployed in a docker-compose file together with the database.

## **API Testing and Validation**:

For testing we made use of pytest, subprocess libs.
Our tests involve:
1. Testing the connection with the lake
2. Testing the API

Note: since we are running the tests during CI we have commented the test_service.py before sending to GH.
This happens because to correctly tests in CI we need to also make user of testcontainers lib and factory boy to mock some data in there, and although this is a very important step for a live example, the timelines did not allowed us to study this approach.