import mlflow
import optuna
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# reader = Reader(rating_scale=(1, 0))
# data = Dataset.load_from_df(data[['userId', 'history', 'read_flag']], reader)


# def objective(trial):
#     n_factors = trial.suggest_int('n_factors', 50, 200)
#     n_epochs = trial.suggest_int('n_epochs', 10, 50)
#     lr_all = trial.suggest_float('lr_all', 0.001, 0.1)
#     reg_all = trial.suggest_int('reg_all', 0.01, 0.1)

#     model = SVD(
#         n_factors=n_factors,
#         n_epochs=n_epochs,
#         lr_all=lr_all,
#         reg_all=reg_all
#     )

#     cv_results = cross_validate(
#         model, data, measures=['RMSE'], cv=3, verbose=False
#     )

#     with mlflow.start_run(nested=True):
#         mlflow.log_params({
#             'n_factors': n_factors,
#             'n_epochs': n_epochs,
#             'lr_all': lr_all,
#             'reg_all': reg_all,
#         })
#         mlflow.log_metrics('mean_rmse', cv_results['test_rmse'].mean())
#         return cv_results['test_rmse'].mean()


# # Set Experiment
# mlflow.set_experiment('Collaborative_Filtering_Optuna')

# # Run
# with mlflow.start_run():
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=20)

# # Log Best parameters and metrics
# best_params = study.best_params
# best_rmse = study.best_value

# mlflow.log_params(best_params)
# mlflow.log_metric('best_rmse', best_rmse)


# # Train Final Model
# best_model = SVD(**best_params)
# trainset = data.build_full_trainset()
# best_model.fit(trainset)

# mlflow.sklearn.log_model(best_model, 'colab_filter_model')


class ColaborativeFilterExperiment:
    def __init__(
        self,
        experiment_name: str,
        data: pd.DataFrame,
        direction: str = 'minimize',
        trials: int = 20,
    ):
        self._experiment_name = experiment_name
        self._direction = direction
        self._data = data
        self._trials = trials

    def objective(self, trial):
        n_factors = trial.suggest_int('n_factors', 50, 200)
        n_epochs = trial.suggest_int('n_epochs', 10, 50)
        lr_all = trial.suggest_float('lr_all', 0.001, 0.1)
        reg_all = trial.suggest_int('reg_all', 0.01, 0.1)

        model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
        )

        cv_results = cross_validate(
            model, self._data, measures=['RMSE'], cv=3, verbose=False
        )

        with mlflow.start_run(nested=True):
            mlflow.log_params({
                'n_factors': n_factors,
                'n_epochs': n_epochs,
                'lr_all': lr_all,
                'reg_all': reg_all,
            })
            mlflow.log_metrics({'mean_rmse': cv_results['test_rmse'].mean()})
            return cv_results['test_rmse'].mean()

    def train_best(self):
        # Train Final Model
        best_model = SVD(**self._best_params)
        trainset = self._data.build_full_trainset()
        best_model.fit(trainset)
        mlflow.sklearn.log_model(best_model, 'colab_filter_model')

    def run_experiment(
        self,
        rating_field: str = 'flag_read',
        rating_a: int = 0,
        rating_b: int = 1,
    ):
        reader = Reader(rating_scale=(rating_a, rating_b))
        self._data = Dataset.load_from_df(
            self._data[['userId', 'history', rating_field]], reader
        )
        # Set Experiment
        mlflow.set_experiment(self._experiment_name)

        # Run
        with mlflow.start_run():
            study = optuna.create_study(direction=self._direction)
            study.optimize(self.objective, n_trials=self._trials)

        # Log Best parameters and metrics
        self._best_params = study.best_params
        self._best_rmse = study.best_value

        mlflow.log_params(self._best_params)
        mlflow.log_metric('best_rmse', self._best_rmse)

        self.train_best()


if __name__ == '__main__':
    path = 'globo_recommendation_fiap/data/challenge_files/local'
    data = pd.read_parquet(f'{path}/acessos_filtrados.parquet')
    train_data = data[['userId', 'history', 'flag_read']]
    train_data_test = train_data.iloc[:10000]
    experiment = 'Teste_classe_surprise'
    ml_experiment = ColaborativeFilterExperiment(
        experiment_name=experiment, data=train_data
    )

    ml_experiment.run_experiment()
