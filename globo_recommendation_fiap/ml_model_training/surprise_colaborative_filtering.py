import mlflow
import optuna
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


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
        n_factors = trial.suggest_int('n_factors', 10, 300)
        n_epochs = trial.suggest_int('n_epochs', 10, 300)
        lr_all = trial.suggest_float('lr_all', 0.001, 0.2)
        # estava como suggest_int
        reg_all = trial.suggest_float('reg_all', 0.01, 0.1)
        # init_mean = trial.suggest_float('init_mean', 0, 0.1)
        # init_std_dev = trial.suggest_float('init_std_dev', 0.01, 0.1)

        model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            # init_mean=init_mean,
            # init_std_dev=init_std_dev,
        )

        cv_results = cross_validate(
            model, self._data, measures=['RMSE', 'MAE'], cv=3, verbose=False
        )

        # Print outras métricas
        # Média do erro absoluto
        print(f'MAE: {cv_results["test_mae"].mean()}')
        # Média do erro quadrático médio
        print(f'RMSE: {cv_results["test_rmse"].mean()}')
        # Média da métrica R2
        # print(f"R2: {cv_results['test_r2'].mean()}")
        # Média do erro percentual absoluto médio
        # print(f"MAPE: {cv_results['test_mape'].mean()}")

        with mlflow.start_run(nested=True):
            mlflow.log_params({
                'n_factors': n_factors,
                'n_epochs': n_epochs,
                'lr_all': lr_all,
                'reg_all': reg_all,
                # 'init_mean': init_mean,
                # 'init_std_dev': init_std_dev,
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
        rating_a: int = 1,
        rating_b: int = 0,
    ):
        reader = Reader(rating_scale=(rating_a, rating_b))
        # self._data = Dataset.load_from_df(self._data[['userId', 'history',
        # rating_field]],reader)
        self._data = Dataset.load_from_df(
            self._data[['User', 'Page', rating_field]], reader
        )
        # Set Experiment
        mlflow.set_experiment(self._experiment_name)

        # Run
        with mlflow.start_run():
            study = optuna.create_study(direction=self._direction)
            study.optimize(self.objective, n_trials=self._trials)

        # Log Best parameters and metrics
        self._best_params = study.best_params
        print('parametros: ', self._best_params)
        self._best_rmse = study.best_value

        mlflow.log_params(self._best_params)
        mlflow.log_metric('best_rmse', self._best_rmse)

        self.train_best()


if __name__ == '__main__':
    path = 'globo_recommendation_fiap/data/challenge_files/local'
    data = pd.read_parquet(f'{path}/acessos_filtrados.parquet')
    train_data = data[['userId', 'history', 'flag_read']]
    train_data_test = train_data.iloc[:10000]
    experiment = 'recomender'
    ml_experiment = ColaborativeFilterExperiment(
        experiment_name=experiment, data=train_data_test
    )

    ml_experiment.run_experiment(
        rating_field='Score',
        rating_a=train_data_test['Score'].max(),
        rating_b=train_data_test['Score'].min(),
    )
