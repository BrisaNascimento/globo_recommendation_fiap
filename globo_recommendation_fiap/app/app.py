from http import HTTPStatus

from fastapi import FastAPI

from globo_recommendation_fiap.app.schemas.schemas import Message

app = FastAPI()


@app.get('/', status_code=HTTPStatus.OK, response_model=Message)
async def read_root():
    return {'message': 'Hello World'}
