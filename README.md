# Globo_Recommendation_FIAP

In this document we are going to provide some initial guidance on how to work with this project.

    1. Environment variables: the enviroment variables must be added into a file .env in the root of this project, every variable regardless where this is used must be added there. We are going to use pydantic settings to consume it.

    2. Taskipy: to facilitate test runs, lint and API run we have implemented taskipy in the dev environmenr, to run any of the available commands you should run the below comand in you shell for eg tests

## Comand Example for Taskipy
```bash
task test
```

## List of comands available for taskipy

* lint: will trigger ruff to validate if your code follows pep8
* format: Will trigger ruff to format your code following pep8 (Note: some errors must be manually fixed such as line length limit) 
* run: Will run the fast api in DEV mode
* pre-test: Will run the lint and format
* test: Will run pytest
* post-test: Will create the coverage file
* docs: Will start the mkdocs server

Note: If you simply run task test taskipy already understands that it must also run pre and post test meaning that all 3 comands are executed with a simple task test comand

## Instructions to Run the project

    1. First of all you need a valid mlflow experiment with a model saved
    2. run task ml_flow_start to start the ml_flow page
    3. Get the experiment name and run id the the valid experiment
    4. Under ml_model_training edit the file register_model to add your experiment name and run id
    5. Execute task register_model (this wil register the ML model in Bento)
    6. Exectute bentoml build (this will build the image based on the bentofile)
    7. Follow the bento instructions to conteinerize the image
    8. Execute task db_up (this will run the docker-compose)
    9. The final result must be a container with DB, PGADMIN and the ML api

NOTE: the project documentation is being created with mkdocs so will be in a different link which will be provided later


