import itertools

import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

from globo_recommendation_fiap.ml_model_training.matrix_generator import (
    MatrixGenerator,
)


class AlsCollaborativeFilter:
    def __init__(
        self,
        data: pd.DataFrame,
        param_grid: dict,
        row: str,
        column: str,
        value: str,
    ) -> None:
        self.raw_data = data
        self.param_grid = param_grid
        self.set_train_data(row=row, column=column, value=value)

    def set_train_data(self, row: str, column: str, value: str) -> coo_matrix:
        train, test = train_test_split(self.raw_data, test_size=0.2)
        train_m_gen = MatrixGenerator(train)
        self.train_data = train_m_gen.get_matrix(
            column=column, row=row, value=value
        )[2].tocsr()
        test_m_gen = MatrixGenerator(test)
        self.test_data = test_m_gen.get_matrix(
            column=column, row=row, value=value
        )[2].tocsr()

    def train(self, k: int = 5) -> None:
        num_users, num_pages = self.train_data.shape
        mlflow.set_experiment('ALS_User_Collaborative_Filter_ALS')
        for params in itertools.product(*self.param_grid.values()):
            factors, regularization, iterations = params
            with mlflow.start_run():
                mlflow.log_param('factors', factors)
                mlflow.log_param('regularization', regularization)
                mlflow.log_param('iterations', iterations)
                model = AlternatingLeastSquares(
                    factors=factors,
                    regularization=regularization,
                    iterations=iterations,
                )
                model.fit(self.train_data)
                sparsity = 1.0 - (
                    self.train_data.count_nonzero() / (num_users * num_pages)
                )
                precision = self.precision_at_k(model, k)
                mlflow.log_metric('sparsity', sparsity)
                mlflow.log_metric(f'precision_at_{k}', precision)
                mlflow.sklearn.log_model(model, 'als_model')

    def precision_at_k(self, model: AlternatingLeastSquares, k: int) -> float:
        """
        Calculates the precision@k metric using separate train and test data.

        Args:
            model: Trained AlternatingLeastSquares model.
            k: Number of top-k recommendations to consider.

        Returns:
            float: Average precision@k across all users in the test data.
        """

        precisions = []
        num_users = self.test_data.shape[0]

        for user_idx in range(num_users):
            # Get known interactions for the user in the test data
            test_interactions = self.test_data[user_idx].toarray().flatten()
            relevant_items = set(np.where(test_interactions > 0)[0])

            if len(relevant_items) == 0:
                continue

            # Get recommendations for the user
            recommended, _ = model.recommend(
                user_idx, self.train_data[user_idx], N=k
            )
            recommended_set = set(recommended)

            # Calculate precision
            precision = len(relevant_items & recommended_set) / k
            precisions.append(precision)

        return sum(precisions) / len(precisions) if precisions else 0


if __name__ == '__main__':
    # data = {
    #     'user_id': [1, 1, 2, 2, 3],
    #     'page_id': [101, 102, 101, 103, 104],
    #     'interaction_score': [5, 3, 4, 2, 1],
    # }
    path = 'globo_recommendation_fiap/data/challenge_files/local'
    data = pd.read_parquet(f'{path}/user_colab_filter.parquet')
    data = data.head(10000)
    row = 'userId'
    column = 'history'
    value = 'engagement_score_pca'
    df = pd.DataFrame(data)
    factors = [10, 20, 50]
    regularization = [0.01, 0.1, 1.0]
    iterations = [10, 30, 50]
    param_grid = {
        'factors': factors,
        'regularization': regularization,
        'iterations': iterations,
    }
    als_train = AlsCollaborativeFilter(
        data=df, param_grid=param_grid, row=row, column=column, value=value
    )

    als_train.train()
