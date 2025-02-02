from datetime import date, timedelta
from typing import List

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommenderMLflow(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load necessary files (e.g., news base) when
        loading model from MLflow
        """
        self.news_base = pd.read_parquet(context.artifacts['news_data'])

    @staticmethod
    def filter_news_by_date(
        news: pd.DataFrame, reference_date: date
    ) -> pd.DataFrame:
        """
        Will receive the news dataset, then filter based on date
        to pass only recent news to cosine similarity calculation.
        Period: D-3 to current
        Input:
            1 - date: a date value in format YYYY-MM-DD
            2 - news: a pandas dataframe with all news base

        Output:
            1 - filtered_news: a pandas dataframe with only with current news
        """
        start_date = reference_date - timedelta(days=3)
        return news[
            (news['date'] >= start_date) & (news['date'] <= reference_date)
        ]

    @staticmethod
    def calculate_cosine_similarity(
        reference_news: np.array, filtered_news: pd.DataFrame, top_k: int = 3
    ) -> pd.DataFrame:
        """
            This function calculate the consise similarity
        between one reference news and the filtered news dataset.
        Both reference news and filtered news dataset calculation
        are based on a 384 dimension embbeding on the body (content)
        of the news article.
        Input:
            1 - reference_news: a 384 dim vector with the reference
        article we want to suggest similar content
            2 - filtered_news: a pandas dataframe with only with current news
            3 - top_k: an int value to detemine the total of
        similar content to suggest
        Output:
            1 - a list containing the top_k similar news
        """
        reference_news = reference_news.reshape(1, -1)

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            np.vstack(filtered_news['content_embbeding']), reference_news
        ).flatten()

        # Add similarity scores to the dataframe
        filtered_news = filtered_news.copy()
        filtered_news['cosine_similarity'] = cosine_similarities

        return filtered_news[['page', 'cosine_similarity']].nlargest(
            top_k, 'cosine_similarity'
        )

    def recommend(
        self, user_id: str, reference_news: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Function responsible for orchestrate the recomendations.
        It will receive a user and a group os reference news for that user,
        then get the recomendations and give it back to the requester.

        Input:
            1 - reference_news: dataframe of news relevant to
        recomend similar content

        output:
            2 - recommendation: dataframe containing the recomendations
        for that user based on similar content
        """
        result: List[pd.DataFrame] = []

        for _, row in reference_news.iterrows():
            page = row['history']
            reference_content = self.news_base.loc[
                self.news_base['page'] == page, 'content_embbeding'
            ].to_numpy()

            if reference_content.size == 0:
                continue  # Skip if no embedding found

            reference_content = reference_content[0]
            filtered_news = self.filter_news_by_date(
                self.news_base, row['date']
            )
            result.append(
                self.calculate_cosine_similarity(
                    reference_content, filtered_news
                )
            )

        recommendations = (
            pd.concat(result)
            if result
            else pd.DataFrame(columns=['page', 'cosine_similarity'])
        )
        return recommendations

    def predict(self, context, model_input):
        """
        This function is compatible with mlflow.
        It will receive a dataset with the user and all news to consider

        Input:
            1 - model_input: DataFrame with news to calculate similarity

        Output:
            1 - tuple: containing userId and the dataframe
        with recommended news
        """
        user_id = model_input['userId'].iloc[0]
        reference_news = model_input[['history', 'date']]
        return self.recommend(user_id, reference_news)
