# Saving

After training the model (Refer to Training section) a pickle file is generated, after the model is generated we then make use of bentoml.

Bentoml allows you to register your model and later with the use of the bentofile.yml it will provide you with the tools to build a docker image which is later used in a docker-compose file.

For more information about bentoml and about the deployment of the model, please check the session [Packaging for deployment](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html)