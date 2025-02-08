import pickle

import mlflow
import numpy as np
import optuna
import pandas as pd

from .content_based_recommender import ContentRecomender


# FLAVIO: You need to replace this with the real data
def get_user_data():
    np.random.seed(42)
    # 100 users, 384-dim embeddings
    user_embeddings = np.random.rand(100, 384)
    # 5 consumed articles per user
    consumed_news = {
        i: np.random.choice(
            500, 5, replace=False) for i in range(100)}
    # 500 news articles, 384-dim embeddings
    news_embeddings = np.random.rand(500, 384)

    news_df = pd.DataFrame({
        'page': np.arange(500),
        'content_embbeding': list(news_embeddings)
    })

    return user_embeddings, consumed_news, news_df


def objective(trial):
    # Tune top_k from 1 to 50
    top_k = trial.suggest_int("top_k", 1, 50)

    user_embeddings, consumed_news, news_df = get_user_data()
    model = ContentRecomender(top_k=top_k)

    all_accuracies = []
    for i, user_embedding in enumerate(user_embeddings):
        recommended = model.predict(
            user_embedding, news_df)["page"].tolist()
        actual = consumed_news[i]

        # Accuracy: Check if recommended news is in consumed news
        accuracy = len(
            set(recommended) & set(actual)) / len(actual)
        all_accuracies.append(accuracy)

    mean_accuracy = np.mean(all_accuracies)

    # Log results to MLflow
    with mlflow.start_run():
        mlflow.log_param("top_k", top_k)
        mlflow.log_metric("accuracy", mean_accuracy)

    return mean_accuracy  # Optuna maximizes this value


def train_final_model(best_top_k: int):
    final_model = ContentRecomender(top_k=best_top_k)
    with open("content_recommender.pkl", "wb") as f:
        pickle.dump(final_model, f)
    # Log the model in MLflow
    with mlflow.start_run():
        mlflow.log_param("top_k", best_top_k)
        mlflow.sklearn.log_model(final_model, "content_recommender")


def run_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    study = optuna.create_study(direction="maximize")  # Maximize accuracy
    study.optimize(objective, n_trials=20)  # Run 20 trials
    print("Best hyperparameters:", study.best_params)
    best_top_k = study.best_params["top_k"]
    train_final_model(best_top_k)


if __name__ == '__main__':
    experiment_name = 'tests_content_base_dummy'
    run_experiment(experiment_name)
