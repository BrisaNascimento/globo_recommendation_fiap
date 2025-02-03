"""Module responsible for serving with BentoML."""

from __future__ import annotations

from typing import List

import bentoml

S_USER = '0004e1ddec9a5d67faa56bb734d733628a7841c10c7255c0c507b7d1d4114f06'
S_ITEM = '43b8e36b-5a0b-4c76-9adf-fb5366dbc330'


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
    """Summarization."""

    bento_model = bentoml.models.get('recomender:latest')

    def __init__(self):
        self.pyfunc_model = bentoml.mlflow.load_model(self.bento_model)
        sklearn_wrapper = self.pyfunc_model._model_impl
        self.model = sklearn_wrapper.sklearn_model

    @bentoml.api(batchable=False)
    def recommend(self, uid: str = S_USER, iid: str = S_ITEM) -> List[str]:
        """Summarize texts.

        Args:
            texts (list[str]): Texts to be summarized.

        Returns:
            list[str]: Summarized texts.
        """

        with bentoml.monitor('Recomendation Model') as mon:
            mon.log([uid, iid], name='request', role='input', data_type='list')
            prediction = self.model.predict(uid, iid)

            # Format the prediction as a list of strings
            recommendation = [
                f'User: {prediction.uid}',
                f'Item: {prediction.iid}',
                f'Estimated Rating: {prediction.est}',
                f'Details: {prediction.details}',
            ]
            mon.log(
                recommendation,
                name='response',
                role='prediction',
                data_type='list',
            )
            return recommendation
