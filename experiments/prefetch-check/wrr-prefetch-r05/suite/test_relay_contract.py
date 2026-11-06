import copy,json,unittest
from pathlib import Path
from relay_contract import assert_relay_ports
ROOT=Path(__file__).resolve().parent
class RelayPortContract(unittest.TestCase):
    def test_frozen_config_matches_all_endpoints(self):
        assert_relay_ports(json.loads((ROOT/'config.json').read_text()),ROOT/'network/reverse_relay.py')
    def test_stale_relay_with_changed_config_is_rejected(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        for field in ('port','grpc_port','relay_channel_port','server_url','specedge_server_url'):
            with self.subTest(field=field):
                broken=copy.deepcopy(cfg)
                broken[field]=cfg[field]+100 if isinstance(cfg[field],int) else cfg[field].rsplit(':',1)[0]+':39999'
                with self.assertRaisesRegex(ValueError,'Config/relay port mismatch'):
                    assert_relay_ports(broken,ROOT/'network/reverse_relay.py')
