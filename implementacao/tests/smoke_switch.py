"""Switch L2 mínimo para o teste de fumaça (o os-ken 4 não traz simple_switch_13)."""
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.lib.packet import ethernet, packet
from os_ken.ofproto import ofproto_v1_3


class SmokeSwitch(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features(self, ev):
        dp = ev.msg.datapath
        p, o = dp.ofproto_parser, dp.ofproto
        self.add_flow(dp, 0, p.OFPMatch(), [p.OFPActionOutput(o.OFPP_CONTROLLER, o.OFPCML_NO_BUFFER)])

    def add_flow(self, dp, prio, match, actions):
        p, o = dp.ofproto_parser, dp.ofproto
        inst = [p.OFPInstructionActions(o.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=prio, match=match, instructions=inst))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        p, o = dp.ofproto_parser, dp.ofproto
        in_port = msg.match["in_port"]
        eth = packet.Packet(msg.data).get_protocol(ethernet.ethernet)
        table = self.mac_to_port.setdefault(dp.id, {})
        table[eth.src] = in_port
        out = table.get(eth.dst, o.OFPP_FLOOD)
        actions = [p.OFPActionOutput(out)]
        if out != o.OFPP_FLOOD:
            self.add_flow(dp, 1, p.OFPMatch(in_port=in_port, eth_dst=eth.dst), actions)
        data = msg.data if msg.buffer_id == o.OFP_NO_BUFFER else None
        dp.send_msg(p.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                                   actions=actions, data=data))
