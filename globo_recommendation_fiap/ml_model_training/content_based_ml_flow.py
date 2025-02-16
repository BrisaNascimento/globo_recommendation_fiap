import mlflow
import numpy as np
import pandas as pd

from .content_based_recommender import ContentRecomender


def get_user_data():
    base_path = 'globo_recommendation_fiap/data/challenge_files/local'
    last_news = pd.read_parquet(f'{base_path}/ultimas_noticias.parquet')
    last_access = pd.read_parquet(f'{base_path}/ultimos_acessos.parquet')
    validacao = pd.read_parquet(f'{base_path}/acessos_val.parquet')

    # Remocao das noticiais que nao estao na base de
    # dados para avaliacao do modelo
    filtered_validation = validacao[
        validacao['history'].isin(last_news['page'])
    ]

    return last_news, last_access, filtered_validation


def call_recomentender_model(top_k: int = -1):
    last_news, last_access, validacao = get_user_data()
    model = ContentRecomender(top_k=top_k)

    in_recommendations = 0
    all_accuracies = []

    # Group last_access by userId for faster lookups
    user_access_groups = last_access.groupby('userId')

    for index, val in validacao.iterrows():
        user_id = val['userId']
        actual = val['history']

        # Get all embeddings for the current user
        if user_id in user_access_groups.groups:
            user_embeddings = user_access_groups.get_group(user_id)[
                'content_embbeding'
            ].values
            user_embeddings = np.vstack(user_embeddings)

            # Batch predict for all embeddings of the current user
            recommended = model.predict_batch(user_embeddings, last_news)[
                'page'
            ].tolist()

            if actual in recommended:
                in_recommendations += 1
                all_accuracies.append(recommended.index(actual) + 1)
            else:
                all_accuracies.append(np.nan)

    accuracy = in_recommendations / len(all_accuracies)

    return accuracy


def run_experiment(experiment_name: str, top_k: int = 10):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        accuracy = call_recomentender_model(top_k=top_k)
        print(f'Accuracy: {accuracy}')
        mlflow.log_metric('accuracy', accuracy)
        model = ContentRecomender(top_k=top_k)
        mlflow.sklearn.log_model(model, artifact_path='content_recommender')


if __name__ == '__main__':
    experiment_name = 'tests_content_base_dummy_2'
    run_experiment(experiment_name=experiment_name, top_k=15)
