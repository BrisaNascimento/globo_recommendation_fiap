# Globo_Recommendation_FIAP

In this document we are going to provide some initial guidance on how to work with this project.

    1. Environment variables: the enviroment variables must be added into a file .env in the root of this project, every variable regardless where this is used must be added there. We are going to use pydantic settings to consume it.

    2 Taskipy: to facilitate test runs, lint and API run we have implemented taskipy in the dev environmenr, to run any of the available commands you should run the below comand in you shell for eg tests

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


