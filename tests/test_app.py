from http import HTTPStatus


def test_root_should_return_Hello_World_and_OK(client):
    response = client.get('/')

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'message': 'Hello World'}
