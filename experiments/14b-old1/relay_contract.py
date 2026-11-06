"""Check the frozen relay endpoints before validation or deployment."""
import ast
from pathlib import Path
from urllib.parse import urlsplit

def assert_relay_ports(config, relay_path):
    constants={}
    for node in ast.parse(Path(relay_path).read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SERVICES','LOCAL_PORTS','CHANNEL_PORT'):
                    constants[target.id]=ast.literal_eval(node.value)
    expected={'SERVICES':{'model':config['grpc_port'],'coordinator':config['port']},
              'LOCAL_PORTS':{'model':int(config['specedge_server_url'].rsplit(':',1)[1]),
                             'coordinator':urlsplit(config['server_url']).port},
              'CHANNEL_PORT':config['relay_channel_port']}
    if constants != expected:
        raise ValueError('Config/relay port mismatch: '+repr({'config':expected,'relay':constants}))
    return constants
