import mlflow
import numpy as np
import optuna
import pandas as pd

from .content_based_recommender import ContentRecomender


def get_user_data():
    base_path = 'globo_recommendation_fiap/data/challenge_files/local'
    last_news = pd.read_parquet(f'{base_path}/ultimas_noticias.parquet')
    last_access = pd.read_parquet(f'{base_path}/ultimos_acessos.parquet')
    validacao = pd.read_parquet(f'{base_path}/acessos_val.parquet')
    # validacao = validacao.head(1000)

    return last_news, last_access, validacao


def objective(trial):
    # Tune top_k from 1 to 50
    top_k = trial.suggest_int('top_k', 1, 50)

    last_news, last_access, validacao = get_user_data()
    model = ContentRecomender(top_k=top_k)
    all_accuracies = []

    # Group last_access by userId for faster lookups
    user_access_groups = last_access.groupby('userId')

    for index, val in validacao.iterrows():
        user_id = val['userId']
        actual = val['history']

        # Get all embeddings for the current user
        if user_id in user_access_groups.groups:
            user_embeddings = user_access_groups.get_group(
                user_id)['content_embbeding'].values
            user_embeddings = np.vstack(user_embeddings)

            # Batch predict for all embeddings of the current user
            recommended = model.predict_batch(
                user_embeddings, last_news)['page'].tolist()
        else:
            recommended = []

        # Calculate accuracy
        accuracy = len(
            set(recommended) & set(actual)) / len(actual) if actual else 0
        all_accuracies.append(accuracy)

    mean_accuracy = np.mean(all_accuracies)

    # Log results to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_param('top_k', top_k)
        mlflow.log_metric('accuracy', mean_accuracy)

    return mean_accuracy  # Optuna maximizes this value


def train_final_model(best_top_k: int):
    final_model = ContentRecomender(top_k=best_top_k)
    with mlflow.start_run():
        mlflow.log_param('top_k', best_top_k)
        mlflow.sklearn.log_model(final_model, 'content_recommender')


def run_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        study = optuna.create_study(direction='maximize')  # Maximize accuracy
        study.optimize(objective, n_trials=20)  # Run 20 trials
        print('Best hyperparameters:', study.best_params)
        best_top_k = study.best_params['top_k']
        mlflow.end_run()
        train_final_model(best_top_k)


if __name__ == '__main__':
    experiment_name = 'tests_content_base_dummy_2'
    run_experiment(experiment_name)
