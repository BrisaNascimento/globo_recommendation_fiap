# Training

The model training has been performed with [ML Flow](https://mlflow.org/)

ML flow is a tool that allows you to perform and store your ML experiments keeping track of the models generated in each row and then optimizing it to find the best possible model.

To perform the optimization, initially we have made use of Optuna, this has been aplied at the initial stages of the REcomender class and also when trying to train the Collaborative Filtering, for more information, refer to [Hyperparameter Tuning with MLflow and Optuna
](https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/index.html).

After training the model we have then registered it using bentoml (refer to Saving sessions).