# Part 2 of Project 3
#
# based on Lab Final from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr, IPAddr6, EthAddr
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.packet.arp import arp

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


class Part2Controller(object):
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

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"], nw_proto=1)
        msg.actions = []
        self.connection.send(msg)
    
        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match() 
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)

    def s2_setup(self):
        # put switch 2 rules here

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"], nw_proto=1) 
        msg.actions = [] 
        self.connection.send(msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)

    def s3_setup(self):
        # put switch 3 rules here

        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"], nw_proto=1)
        msg.actions = []
        self.connection.send(msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match()
        allow_msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))
        self.connection.send(allow_msg)

    def cores21_setup(self):
        self.ip_mac_mapping = {}

    def dcs31_setup(self):
        # put datacenter switch rules here
        
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match(dl_type=0x800, nw_src=IPS["hnotrust"])
        msg.actions = []  
        self.connection.send(msg)

        allow_msg = of.ofp_flow_mod()
        allow_msg.match = of.ofp_match() 
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
        packet = event.parsed  
        if not packet.parsed:
            print("Ignoring incomplete packet")
            return

        packet_in = event.ofp  # The actual ofp_packet_in message.

        if packet.type != 0x86DD:
            log.info("这有一个没处理的新包")
            log.info(packet)
            log.info(packet.payload)

        if packet.type == ethernet.ARP_TYPE:
            arp_packet = packet.payload
            log.info("这是一个ARP包")   
            self.update_mac_ip_mapping(arp_packet.protosrc, arp_packet.hwsrc, event.port)
            if arp_packet.opcode == arp.REQUEST:
                self.handle_arp_response(arp_packet, event)

    def update_mac_ip_mapping(self, ip, mac,port):
        if ip not in self.ip_mac_mapping:
            self.ip_mac_mapping[ip] = {'port': port, 'mac': mac}
            self.update_flow_table(ip, mac, port)
            log.info(f"更新： IP-MAC mapping: {ip} -> mac {mac} port {port}")

    def update_flow_table(self, ip, mac, port):
        msg = of.ofp_flow_mod()

        msg.match = of.ofp_match(dl_type=0x800, nw_dst=IPAddr(ip))

        action1 = of.ofp_action_dl_addr.set_dst(mac) 
        msg.actions.append(action1)

        action2 = of.ofp_action_dl_addr.set_src(self.connection.ports[port].hw_addr) 
        msg.actions.append(action2)
    
        action = of.ofp_action_output(port=port)
        msg.actions.append(action)
        self.connection.send(msg)
        log.info(f"添加表项Flow rule installed for {ip} -> on port {port}")
        log.info(self.ip_mac_mapping)

    def handle_arp_response(self, arp_packet, event):
        port = event.port
        mac = event.connection.ports[port].hw_addr
        arp_reply = arp()
        arp_reply.opcode = arp.REPLY
        arp_reply.hwsrc = mac
        arp_reply.hwdst = arp_packet.hwsrc
        arp_reply.protosrc = arp_packet.protodst
        arp_reply.protodst = arp_packet.protosrc

        eth = ethernet(type=ethernet.ARP_TYPE)
        eth.src = mac 
        eth.dst = arp_packet.hwsrc
        eth.payload = arp_reply

        packet_out = eth.pack()

        msg = of.ofp_packet_out()
        msg.data = packet_out
        action = of.ofp_action_output(port=port)
        msg.actions.append(action)


        self.connection.send(msg)

        log.info(f"ARP回复已发送：{arp_reply}")
        log.info("ARP答复")
        log.info(arp_reply)

def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Part2Controller(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
