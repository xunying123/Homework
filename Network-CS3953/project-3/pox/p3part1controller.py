# Part 1 of Project 3
#
# based on Lab Final from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr, IPAddr6, EthAddr

log = core.getLogger()

# Convenience mappings of hostnames to ips
IPS = {
    "h10": "10.0.1.10",
    "h20": "10.0.2.20",
    "h30": "10.0.3.30",
    "serv1": "10.0.4.10",
    "hnotrust": "172.16.10.100",
}

# Convenience mappings of hostnames to subnets
SUBNETS = {
    "h10": "10.0.1.0/24",
    "h20": "10.0.2.0/24",
    "h30": "10.0.3.0/24",
    "serv1": "10.0.4.0/24",
    "hnotrust": "172.16.10.0/24",
}


class Part1Controller(object):
    """
    A Connection object for that switch is passed to the __init__ function.
    """

    def __init__(self, connection):
        print(connection.dpid)
        # Keep track of the connection to the switch so that we can
        # send it messages!
        self.connection = connection

        # This binds our PacketIn event listener
        connection.addListeners(self)
        # use the dpid to figure out what switch is being created
        if connection.dpid == 1:
            self.s1_setup()
        elif connection.dpid == 2:
            self.s2_setup()
        elif connection.dpid == 3:
            self.s3_setup()
        elif connection.dpid == 21:
            self.cores21_setup()
        elif connection.dpid == 31:
            self.dcs31_setup()
        else:
            print("UNKNOWN SWITCH")
            exit(1)

    def s1_setup(self):
        # put switch 1 rules here
        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()  # 匹配所有流量
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)


    def s2_setup(self):
        # put switch 2 rules here
        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()  # 匹配所有流量
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)

    def s3_setup(self):
        # put switch 3 rules here
        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()  # 匹配所有流量
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)

    def cores21_setup(self):
        # put core switch rules here
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"], nw_proto=1)
        msg.actions = []
        self.connection.send(msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPS["h10"]) 
        allow_msg.actions.append(of.ofp_action_output(port=1))
        self.connection.send(allow_msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPS["h20"])
        allow_msg.actions.append(of.ofp_action_output(port=2))
        self.connection.send(allow_msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPS["h30"]) 
        allow_msg.actions.append(of.ofp_action_output(port=3))
        self.connection.send(allow_msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPS["serv1"])
        allow_msg.actions.append(of.ofp_action_output(port=4))
        self.connection.send(allow_msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPS["hnotrust"])
        allow_msg.actions.append(of.ofp_action_output(port=5))
        self.connection.send(allow_msg)

    def dcs31_setup(self):
        # put datacenter switch rules here
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"])
        msg.actions = []  # 丢弃所有来自 hnotrust 的包
        self.connection.send(msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()  # 匹配所有流量
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)


    # used in part 2 to handle individual ARP packets
    # not needed for part 1 (USE RULES!)
    # causes the switch to output packet_in on out_port
    def resend_packet(self, packet_in, out_port):
        msg = of.ofp_packet_out()
        msg.data = packet_in
        action = of.ofp_action_output(port=out_port)
        msg.actions.append(action)
        self.connection.send(msg)

    def _handle_PacketIn(self, event):
        """
        Packets not handled by the router rules will be
        forwarded to this method to be handled by the controller
        """

        packet = event.parsed  # This is the parsed packet data.
        if not packet.parsed:
            log.warning("Ignoring incomplete packet")
            return

        packet_in = event.ofp  # The actual ofp_packet_in message.
        print(
            "Unhandled packet from " + str(self.connection.dpid) + ":" + packet.dump()
        )

        log.info(event.port)


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Part1Controller(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
