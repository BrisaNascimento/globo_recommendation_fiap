# Project Requirements

This project has been built using poetry as the dependency management tool and
this recommendation system process has as requirements the below libs:

1. **pandas = "^2.2.3"**
2. **pydantic = "^2.10.6"**
3. **pydantic-settings = "^2.7.1"**
4. **scikit-learn = "^1.6.1"**
5. **scipy = "^1.15.1"**
6. **mlflow = "^2.20.1"**
7. **optuna = "^4.2.0"**
8. **scikit-surprise = "^1.1.4"**
9. **numpy = "<2.0"**
10. **pyarrow = "<19.0"**
11. **fastapi = {extras = ["standard"], version = "^0.115.8"}**
12. **azure-storage-blob = "^12.24.1"**
13. **bentoml = "^1.3.21"**
14. **sqlalchemy = "^2.0.38"**
15. **psycopg = {extras = ["binary"], version = "^3.2.4"}**

So in order to install the packages to use the service you can run the command:

```bash
poetry install --only prod
```