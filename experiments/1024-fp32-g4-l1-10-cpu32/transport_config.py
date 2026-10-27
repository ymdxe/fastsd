"""Keep HTTP connections alive across CPU drafting gaps, without POST retries."""
import uvicorn

HTTP_KEEP_ALIVE_SECONDS=7200


def http_server_config(app,**kwargs):
    kwargs.setdefault('timeout_keep_alive',HTTP_KEEP_ALIVE_SECONDS)
    return uvicorn.Config(app,**kwargs)
