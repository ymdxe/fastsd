"""Public-fixture, no-model framing validation on ephemeral localhost sockets."""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,io,json,socket,struct,threading,time,unittest
from pathlib import Path
SOURCE=Path(__file__).with_name("reverse_relay.py")
spec=importlib.util.spec_from_file_location("framed_relay_under_test",SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
PUBLIC=b"public-framing-tests-not-a-deployed-authentication-key"


def until(condition,timeout=5):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  if condition():return True
  time.sleep(.01)
 return bool(condition())


def read_eof(sock,slow=False):
 output=bytearray()
 while True:
  data=sock.recv(8192 if slow else 65536)
  if not data:return bytes(output)
  output.extend(data)
  if slow:time.sleep(.0001)


def data_frame(payload):return r.FRAME_HEADER.pack(r.FRAME_DATA,len(payload))+payload
EOF=r.FRAME_HEADER.pack(r.FRAME_EOF,0)


class FramingTests(unittest.TestCase):
 def test_fragmented_header_payload_and_multiple_frames(self):
  payload=b"fragment-test-"*113
  wire=data_frame(payload)+data_frame(b"second")+EOF
  decoder=r.FrameDecoder();output=bytearray()
  for byte in wire:decoder.feed(bytes([byte]),output)
  self.assertEqual(output,payload+b"second");self.assertTrue(decoder.eof)
  decoder.transport_eof()

 def test_maximum_frame_and_bounded_combined_input(self):
  payload=b"m"*r.MAX_FRAME_SIZE
  decoder=r.FrameDecoder();output=bytearray();wire=data_frame(payload)+EOF
  for offset in range(0,len(wire),7331):decoder.feed(wire[offset:offset+7331],output)
  self.assertEqual(output,payload);self.assertTrue(decoder.eof)
  with self.assertRaises(r.FrameProtocolError):r.FrameDecoder().feed(b"x",bytearray(r.BUFFER_LIMIT))

 def test_malformed_oversized_duplicate_and_post_eof_frames(self):
  cases=[r.FRAME_HEADER.pack(99,0),r.FRAME_HEADER.pack(r.FRAME_DATA,0),r.FRAME_HEADER.pack(r.FRAME_DATA,r.MAX_FRAME_SIZE+1),r.FRAME_HEADER.pack(r.FRAME_EOF,1),EOF+EOF,EOF+b"x"]
  for wire in cases:
   with self.subTest(wire=wire[:5].hex()),self.assertRaises(r.FrameProtocolError):r.FrameDecoder().feed(wire,bytearray())
  decoder=r.FrameDecoder();decoder.feed(EOF,bytearray())
  with self.assertRaises(r.FrameProtocolError):decoder.feed(data_frame(b"later"),bytearray())
  for wire in (b"",b"\x01",r.FRAME_HEADER.pack(r.FRAME_DATA,5)+b"ab",data_frame(b"no-eof")):
   decoder=r.FrameDecoder();decoder.feed(wire,bytearray())
   with self.assertRaises(r.FrameProtocolError):decoder.transport_eof()

 def make_pair(self):
  left,raw_left=socket.socketpair();channel_left,channel_right=socket.socketpair();raw_right,right=socket.socketpair()
  for sock in(left,right):sock.settimeout(15)
  first=r.Runtime(idle_timeout=15);second=r.Runtime(idle_timeout=15)
  workers=[first.spawn(first.pump,first.add(raw_left),first.add(channel_left)),second.spawn(second.pump,second.add(raw_right),second.add(channel_right))]
  return left,right,first,second,workers

 def assert_drained(self,runtimes,workers):
  for worker in workers:worker.join(5);self.assertFalse(worker.is_alive())
  for runtime in runtimes:
   snapshot=runtime.snapshot()
   self.assertEqual(snapshot.get("active_connections",0),0)
   self.assertEqual(snapshot.get("completed_connections",0),1)
   self.assertEqual(snapshot.get("relay_errors",0),0)
   self.assertEqual(snapshot.get("idle_timeouts",0),0)
   self.assertLessEqual(snapshot.get("outbound_buffer_peak",0),r.BUFFER_LIMIT)
   self.assertLessEqual(snapshot.get("inbound_buffer_peak",0),r.BUFFER_LIMIT)

 def test_halfclose_request_then_large_reverse_response(self):
  left,right,first,second,workers=self.make_pair()
  response=b"reverse-after-eof-"*400000
  errors=[]
  def backend():
   try:
    self.assertEqual(read_eof(right),b"request-before-halfclose")
    time.sleep(.1)
    right.sendall(response);right.shutdown(socket.SHUT_WR)
   except Exception as exc:errors.append(exc)
  thread=threading.Thread(target=backend);thread.start()
  try:
   left.sendall(b"request-before-halfclose");left.shutdown(socket.SHUT_WR)
   self.assertEqual(read_eof(left,slow=True),response)
   thread.join(5);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
   self.assert_drained((first,second),workers)
  finally:left.close();right.close();first.shutdown();second.shutdown();thread.join(1)

 def test_simultaneous_large_bidirectional_traffic_with_slow_readers(self):
  left,right,first,second,workers=self.make_pair()
  payload_left=b"aBcD0123456789"*470000
  payload_right=b"ZYX9876543210"*570000
  received={};errors=[]
  def send(sock,payload):
   try:sock.sendall(payload);sock.shutdown(socket.SHUT_WR)
   except Exception as exc:errors.append(exc)
  def receive(sock,name):
   try:received[name]=read_eof(sock,slow=True)
   except Exception as exc:errors.append(exc)
  threads=[threading.Thread(target=send,args=(left,payload_left)),threading.Thread(target=send,args=(right,payload_right)),threading.Thread(target=receive,args=(left,"left")),threading.Thread(target=receive,args=(right,"right"))]
  try:
   for thread in threads:thread.start()
   for thread in threads:thread.join(20);self.assertFalse(thread.is_alive())
   self.assertEqual(errors,[])
   self.assertEqual(received["left"],payload_right);self.assertEqual(received["right"],payload_left)
   self.assert_drained((first,second),workers)
  finally:
   left.close();right.close();first.shutdown();second.shutdown()
   for thread in threads:thread.join(1)

 def test_logical_eof_finishes_while_peer_transport_stays_open(self):
  client,raw=socket.socketpair();channel,peer=socket.socketpair()
  client.settimeout(5);peer.settimeout(5)
  runtime=r.Runtime(idle_timeout=10)
  worker=runtime.spawn(runtime.pump,runtime.add(raw),runtime.add(channel))
  try:
   client.sendall(b"request");client.shutdown(socket.SHUT_WR)
   decoder=r.FrameDecoder();request=bytearray()
   while not decoder.eof:decoder.feed(peer.recv(4096),request)
   self.assertEqual(request,b"request")
   peer.sendall(data_frame(b"response")+EOF)
   # Deliberately keep peer open: completion must not wait for its TCP FIN.
   self.assertEqual(read_eof(client),b"response")
   worker.join(2);self.assertFalse(worker.is_alive())
   self.assertEqual(runtime.snapshot().get("completed_connections"),1)
  finally:client.close();peer.close();runtime.shutdown()

 def test_peer_eof_keeps_reverse_direction_until_local_eof(self):
  client,raw=socket.socketpair();channel,peer=socket.socketpair()
  client.settimeout(5);peer.settimeout(5)
  runtime=r.Runtime(idle_timeout=10);worker=runtime.spawn(runtime.pump,runtime.add(raw),runtime.add(channel))
  try:
   peer.sendall(EOF);self.assertEqual(read_eof(client),b"")
   self.assertTrue(worker.is_alive())
   client.sendall(b"valid-after-receive-eof");client.shutdown(socket.SHUT_WR)
   decoder=r.FrameDecoder();output=bytearray()
   while not decoder.eof:decoder.feed(peer.recv(4096),output)
   self.assertEqual(output,b"valid-after-receive-eof")
   worker.join(2);self.assertFalse(worker.is_alive())
   self.assertEqual(runtime.snapshot().get("relay_errors",0),0)
  finally:client.close();peer.close();runtime.shutdown()

 def test_truncated_channel_is_error_and_releases_resources(self):
  for wire in (b"",b"\x01",r.FRAME_HEADER.pack(r.FRAME_DATA,10)+b"abc",data_frame(b"data-without-eof")):
   client,raw=socket.socketpair();channel,peer=socket.socketpair();runtime=r.Runtime(idle_timeout=10)
   worker=runtime.spawn(runtime.pump,runtime.add(raw),runtime.add(channel))
   try:
    if wire:peer.sendall(wire)
    peer.close();worker.join(2)
    self.assertFalse(worker.is_alive())
    snapshot=runtime.snapshot()
    self.assertEqual(snapshot.get("protocol_errors"),1)
    self.assertEqual(snapshot.get("completed_connections",0),0)
    self.assertEqual(snapshot.get("active_connections"),0)
    self.assertEqual(snapshot.get("open_sockets"),0)
   finally:client.close();peer.close();runtime.shutdown()

 def test_late_post_eof_data_is_rejected(self):
  client,raw=socket.socketpair();channel,peer=socket.socketpair();client.settimeout(3)
  runtime=r.Runtime(idle_timeout=10);worker=runtime.spawn(runtime.pump,runtime.add(raw),runtime.add(channel))
  try:
   peer.sendall(EOF);self.assertEqual(read_eof(client),b"")
   peer.sendall(data_frame(b"illegal"));worker.join(2)
   self.assertFalse(worker.is_alive());self.assertEqual(runtime.snapshot().get("protocol_errors"),1)
  finally:client.close();peer.close();runtime.shutdown()


class Backend:
 def __init__(self):
  self.listener=socket.socket();self.listener.bind(("127.0.0.1",0));self.listener.listen(32);self.listener.settimeout(.1)
  self.port=self.listener.getsockname()[1];self.stop=threading.Event();self.mode="http";self.sockets=set();self.lock=threading.Lock();self.workers=[]
  self.accept_thread=threading.Thread(target=self.accept,daemon=True);self.accept_thread.start()
 def accept(self):
  while not self.stop.is_set():
   try:sock,_=self.listener.accept()
   except socket.timeout:continue
   except OSError:return
   with self.lock:self.sockets.add(sock)
   thread=threading.Thread(target=self.handle,args=(sock,self.mode),daemon=True);self.workers.append(thread);thread.start()
 def handle(self,sock,mode):
  try:
   sock.settimeout(5);request=bytearray()
   while b"\r\n\r\n" not in request:
    data=sock.recv(4096)
    if not data:return
    request.extend(data)
   if mode=="http":sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
  except OSError:pass
  finally:
   sock.close()
   with self.lock:self.sockets.discard(sock)
 def close(self):
  self.stop.set();self.listener.close()
  with self.lock:connections=list(self.sockets)
  for sock in connections:
   try:sock.shutdown(socket.SHUT_RDWR)
   except OSError:pass
   sock.close()
  self.accept_thread.join(2)
  for worker in self.workers:worker.join(2)


class HubAgentTests(unittest.TestCase):
 def test_http_closure_reconnect_and_resource_recovery(self):
  backend=Backend();hub=r.Hub(PUBLIC,"127.0.0.1",channel_port=0,local_ports={"model":0,"coordinator":0},idle_timeout=10);agent=None
  try:
   hub.start();agent=r.Agent(PUBLIC,"127.0.0.1",channel_port=hub.listeners["channels"][1],backend_ports={"model":backend.port,"coordinator":backend.port},channels=1,control_channels=4,idle_timeout=10);agent.start()
   self.assertTrue(until(lambda:hub.available["coordinator"].qsize()==4))
   for mode in("http","close_without_response"):
    backend.mode=mode
    for _ in range(24):
     client=socket.create_connection(hub.listeners["coordinator"],timeout=5)
     try:
      client.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");response=read_eof(client)
      self.assertEqual(response.endswith(b"OK"),mode=="http")
     finally:client.close()
    self.assertTrue(until(lambda:hub.snapshot().get("active_connections",0)==0 and agent.snapshot().get("active_connections",0)==0 and hub.available["coordinator"].qsize()==4))
   self.assertEqual(hub.snapshot()["active_threads"],3)
   self.assertEqual(agent.snapshot()["active_threads"],5)
   self.assertEqual(hub.snapshot()["open_sockets"],8)
   self.assertEqual(agent.snapshot()["open_sockets"],5)
   self.assertEqual(hub.snapshot().get("protocol_errors",0),0)
   self.assertEqual(agent.snapshot().get("protocol_errors",0),0)
   self.assertGreaterEqual(hub.snapshot()["threads_reaped"],96)
  finally:
   if agent:agent.shutdown()
   hub.shutdown();backend.close()

 def test_old_protocol_and_bad_auth_rejected(self):
  hub=r.Hub(PUBLIC,"127.0.0.1",channel_port=0,local_ports={"model":0,"coordinator":0},idle_timeout=10)
  try:
   hub.start()
   for version,valid_digest in((1,True),(2,False),(None,False)):
    peer=socket.create_connection(hub.listeners["channels"],timeout=3)
    try:
     challenge=json.loads(r.recv_line(peer));self.assertEqual(challenge["version"],2)
     digest=r.digest(PUBLIC,challenge["nonce"],"coordinator") if valid_digest else "0"*64
     response = [] if version is None else {"version":version,"service":"coordinator","digest":digest}
     peer.sendall(json.dumps(response).encode()+b"\n")
     self.assertEqual(peer.recv(1),b"")
    finally:peer.close()
   self.assertTrue(until(lambda:hub.snapshot().get("authentication_failures",0)==2 and hub.snapshot().get("channel_rejections",0)==1))
   self.assertEqual(hub.available["coordinator"].qsize(),0)
  finally:hub.shutdown()


 def test_nondict_challenge_retries_without_losing_agent_slots(self):
  listener=socket.socket();listener.bind(("127.0.0.1",0));listener.listen(8);listener.settimeout(.1)
  stop=threading.Event()
  def send_bad_challenges():
   while not stop.is_set():
    try:peer,_=listener.accept()
    except socket.timeout:continue
    except OSError:return
    try:peer.sendall(b"[]\n")
    finally:peer.close()
  server=threading.Thread(target=send_bad_challenges,daemon=True);server.start()
  agent=r.Agent(PUBLIC,"127.0.0.1",channel_port=listener.getsockname()[1],channels=1,control_channels=1,idle_timeout=10)
  try:
   agent.start()
   self.assertTrue(until(lambda:agent.snapshot().get("channel_reconnects",0)>=2))
   self.assertEqual(agent.snapshot()["active_threads"],2)
  finally:agent.shutdown();stop.set();listener.close();server.join(2)


if __name__=="__main__":
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output");args=parser.parse_args()
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(FramingTests),unittest.defaultTestLoader.loadTestsFromTestCase(HubAgentTests)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful(),"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"protocol_version":r.PROTOCOL_VERSION,"max_frame_bytes":r.MAX_FRAME_SIZE,"buffer_limit_bytes":r.BUFFER_LIMIT,"models_loaded":0,"private_auth_read":False,"public_fixture":True,"network_scope":"localhost ephemeral socket pairs/ports only; cleaned in finally"}
 if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
