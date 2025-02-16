"""Module responsible for serving with BentoML."""

from __future__ import annotations

import json

import bentoml
import numpy as np
import pandas as pd

from globo_recommendation_fiap.utils.db_connect import connect_to_db

S_USER = '0004e1ddec9a5d67faa56bb734d733628a7841c10c7255c0c507b7d1d4114f06'


@bentoml.service(
    resources={'cpu': '4'},
    traffic={'timeout': 10},
    monitoring={'enabled': True},
    metrics={
        'enabled': True,
        'namespace': 'bentoml_service',
    },
)
class Recommender:
    """Content based recommender."""

    bento_model = bentoml.models.get('final_model:latest')

    def __init__(self):
        self.pyfunc_model = bentoml.mlflow.load_model(self.bento_model)
        sklearn_wrapper = self.pyfunc_model._model_impl
        self.model = sklearn_wrapper.sklearn_model
        # self.last_news = download_from_adls(
        #     Settings().CONTAINER_NAME, Settings().LAST_NEWS
        # )

    @bentoml.api(batchable=False)
    def recommend(self, user: str = S_USER) -> str:
        """
        Make documentation
        """

        engine = connect_to_db()
        query = f'''
            SELECT "userId", history, content_embbeding
            FROM user_last_access
            WHERE "userId" = '{user}'
        '''
        query_last_news = '''
            SELECT *
            FROM last_news
        '''
        query_rank = '''
            SELECT *
            FROM last_news_ranking
        '''
        with engine.connect() as connection:
            user_base = pd.read_sql(query, connection)
            user_base['content_embbeding'] = user_base['content_embbeding'].\
                apply(lambda x: np.array([float(i) for i in x.split(',')]))
            last_news = pd.read_sql(query_last_news, connection)
            last_news['content_embbeding'] = last_news['content_embbeding'].\
                apply(lambda x: np.array([float(i) for i in x.split(',')]))

        with bentoml.monitor('Recomendation Model') as mon:
            mon.log(user, name='request', role='input', data_type='list')
            preds = []
            if len(user_base) > 0:
                for index, row in user_base.iterrows():
                    embbeding = row['content_embbeding'].reshape(1, -1)
                    prediction = self.model.predict(embbeding, last_news)
                    preds.append(prediction)
            else:
                columns = ['page', 'rank']
                with engine.connect() as connection:
                    ranked_news = pd.read_sql(query_rank, connection)
                ranked_news.columns = columns
                ranked_news['cossine_similarity'] = 0
                ranked_news = ranked_news[['page', 'cossine_similarity']].\
                    head(15)
                preds.append(ranked_news)
            result = pd.concat(preds)
            result = result[~result['page'].isin(user_base['history'])]

            output_json = {
                'userId': user,
                'preds': result.to_dict(orient='records'),
            }

            json_string = json.dumps(output_json, indent=4, ensure_ascii=False)

            mon.log(
                json_string,
                name='response',
                role='prediction',
                data_type='str',
            )
            return json_string
