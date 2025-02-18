import bentoml


class Register:
    """Register."""

    def __init__(self, title: str):
        """Initialize Register.

        Args:
            title (str): Experiment title.
        """
        self.title = title

    def register_model(self, run_id: str):
        """Register model in mlflow and bentoml.

        Args:
            run_id (str): Run id.
        """
        model_uri = f'runs:/{run_id}/content_recommender'
        # mlflow.register_model(
        #     model_uri,
        #     name=self.title,
        #     tags={"status": "demo", "owner": "julio-bernardes"},
        # )
        bentoml.mlflow.import_model(self.title, model_uri)


if __name__ == '__main__':
    # content_recommender or colab_filter_model
    model_path = 'content_recommender'
    experiment_name = 'final_model'
    run_id = '0408237f72474d7a9c86aeb53848dcbb'
    reg = Register(experiment_name)
    reg.register_model(run_id)
