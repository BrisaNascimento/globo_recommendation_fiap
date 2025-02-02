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
        model_uri = f'runs:/{run_id}/colab_filter_model'
        # mlflow.register_model(
        #     model_uri,
        #     name=self.title,
        #     tags={"status": "demo", "owner": "julio-bernardes"},
        # )
        bentoml.mlflow.import_model(self.title, model_uri)


if __name__ == '__main__':
    experiment_name = 'recomender'
    run_id = 'da20ee974e4d403b8bdc4fbdf006db3f'
    reg = Register(experiment_name)
    reg.register_model(run_id)
