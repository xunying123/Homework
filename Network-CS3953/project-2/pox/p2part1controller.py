# Part 1 of Project 2
#
# based on Lab 4 from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.packet.ethernet import ethernet
from pox.lib.packet.ipv4 import ipv4

log = core.getLogger()


class Firewall(object):
    """
    A Firewall object is created for each switch that connects.
    A Connection object for that switch is passed to the __init__ function.
    """

    def __init__(self, connection):
        # Keep track of the connection to the switch so that we can
        # send it messages!
        self.connection = connection

        # This binds our PacketIn event listener
        connection.addListeners(self)

        self.allow_icmp()

        self.allow_arp()

        self.drop_all()

    def allow_icmp(self):
        # 创建ICMP流表项
        msg = of.ofp_flow_mod()
        msg.priority = 1000  # 设置较高优先级
        msg.match.dl_type = 0x0800  # 以太网协议类型：IPv4
        msg.match.nw_proto = ipv4.ICMP_PROTOCOL  # 网络层协议：ICMP
        msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))  # 正常转发
        self.connection.send(msg)

    def allow_arp(self):
        msg = of.ofp_flow_mod()
        msg.priority = 100  # 设置与ICMP规则相同的优先级
        msg.match.dl_type = 0x0806  # 以太网协议类型：ARP
        msg.actions.append(of.ofp_action_output(port=of.OFPP_NORMAL))  # 正常转发
        self.connection.send(msg)


    def drop_all(self):
        # 默认丢弃其他所有数据包
        drop_msg = of.ofp_flow_mod()
        drop_msg.priority = 10  # 低优先级，确保只在没有匹配其他规则时生效
        self.connection.send(drop_msg)


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
        print("Unhandled packet :" + str(packet.dump()))


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Firewall(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
