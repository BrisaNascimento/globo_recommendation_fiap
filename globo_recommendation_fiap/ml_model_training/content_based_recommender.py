import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecomender:
    def __init__(self, top_k: int = 3):
        """
        Input:
        1 - top_k: an int value to detemine the total
        of similar content to suggest

        """
        self.top_k = top_k

    def calculate_cosine_similarity(
        self, reference_embbeding: np.array, news: pd.DataFrame
    ):
        """
        This function calculate the consise similarity
        between one reference news and the filtered news dataset.
        Both reference news and filtered news dataset calculation are
        based on a 384 dimension embbeding on the body (content) of
        the news article.
        Input:
            1 - reference_embbeding: a 384 dim vector with the reference
        article we want to suggest similar content with shape (1, -1)
            2 - news: a pandas dataframe with only current news
        Output:
            1 - a list containing the top_k similar news
        """

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            np.vstack(news['content_embbeding']), reference_embbeding
        ).flatten()

        # Create a new column in the DataFrame
        news['cosine_similarity'] = cosine_similarities

        recommended_articles = news.sort_values(
            by='cosine_similarity', ascending=False
        )
        return recommended_articles[['page', 'cosine_similarity']].head(
            self.top_k
        )

    def predict(
        self, reference_embbeding: pd.DataFrame, recent_news: pd.DataFrame
    ):
        """
        Function responsible for orchestrate the recomendations.
        It will receive a user and a group os reference news for that user,
        then get the recomendations and give it back to the requester.

        Input:
            1 - reference_embbeding: a numpy array with the embbeding of
        the news to be used as reference for the similarity recommender

        output:
            2 - recommendation: dataframe containing the recomendations
        for that user based on similar content.
        """
        reference_embbeding = reference_embbeding.reshape(1, -1)

        return self.calculate_cosine_similarity(
            reference_embbeding, recent_news
        )

    # added later

    def predict_batch(
        self, embeddings: np.array, recent_news: pd.DataFrame
    ):
        """
        Batch prediction for multiple embeddings.
        Input:
            1 - embeddings: a 2D numpy array of shape (n_embeddings, 384)
            2 - recent_news: a DataFrame containing recent news articles
        Output:
            1 - recommendations: a DataFrame containing all recommendations
        for all embeddings, with columns ['page', 'cosine_similarity']
        """
        # Stack all embeddings into a single 2D array
        embeddings = np.vstack(embeddings)

        # Calculate cosine similarity for all embeddings at once
        cosine_similarities = cosine_similarity(
            np.vstack(recent_news['content_embbeding']), embeddings
        )

        # Create a DataFrame to store all recommendations
        all_recommendations = []

        # For each embedding, get the top_k recommendations
        for i in range(cosine_similarities.shape[1]):
            recent_news['cosine_similarity'] = cosine_similarities[:, i]
            recommended_articles = recent_news.sort_values(
                by='cosine_similarity', ascending=False
            ).head(self.top_k)
            all_recommendations.append(
                recommended_articles[['page', 'cosine_similarity']])

        # Concatenate all recommendations into a single DataFrame
        return pd.concat(all_recommendations, ignore_index=True)
