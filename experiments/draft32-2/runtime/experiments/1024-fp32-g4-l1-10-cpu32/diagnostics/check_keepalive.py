"""One critical transport check: reuse the same connection after the old expiry."""
import http.client
import json
from pathlib import Path
import select
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from transport_config import http_server_config
import uvicorn


async def app(scope,receive,send):
    while (await receive()).get('more_body',False):pass
    body=json.dumps({'client_port':scope['client'][1]}).encode()
    await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
    await send({'type':'http.response.body','body':body})


config=http_server_config(app,host='127.0.0.1',port=0,lifespan='off',log_level='error')
server=uvicorn.Server(config)
thread=threading.Thread(target=server.run,daemon=True);thread.start()
connection=None
try:
    deadline=time.monotonic()+10
    while not server.started:
        if time.monotonic()>deadline:raise TimeoutError('HTTP transport fixture startup')
        time.sleep(.02)
    port=server.servers[0].sockets[0].getsockname()[1]
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
    connection.request('POST','/verify',body=b'{}',headers={'Content-Type':'application/json'})
    first=json.loads(connection.getresponse().read());sock=connection.sock
    start=time.monotonic();time.sleep(6.2)
    assert not select.select([sock],[],[],0)[0],'Server closed the idle connection'
    connection.request('POST','/verify',body=b'{}',headers={'Content-Type':'application/json'})
    second=json.loads(connection.getresponse().read())
    assert connection.sock is sock and first['client_port']==second['client_port']
    result={'ok':True,'idle_s':time.monotonic()-start,'timeout_keep_alive_s':config.timeout_keep_alive,'same_tcp_connection':True,'model_smokes':0}
    (ROOT/'diagnostics/keepalive-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HTTP_KEEPALIVE_CHECK_OK',json.dumps(result),flush=True)
finally:
    if connection:connection.close()
    server.should_exit=True;thread.join(timeout=10)
