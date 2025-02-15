import subprocess

import bentoml

from service import S_USER


def test_summarization_service_integration():
    with subprocess.Popen([
        'bentoml',
        'serve',
        'service:Recommender',
        '-p',
        '50001',
    ]) as server_proc:
        try:
            client = bentoml.SyncHTTPClient(
                'http://localhost:50001', server_ready_timeout=30
            )
            recommendation = client.recommend(user=S_USER)
            assert recommendation
            print(len(recommendation))
            print(len(S_USER))

        finally:
            server_proc.terminate()
