# Description: basic controller module to balance the load of a single switch; Acknowledgement: Marcos Canales Mayo

from pox.core import core
import pox.lib.packet as libpacket
from pox.lib.packet.ethernet import ethernet
from pox.lib.packet.arp import arp
from pox.lib.addresses import IPAddr, EthAddr
import pox.openflow.libopenflow_01 as of
from pox.openflow.of_json import *
import random
import threading

log = core.getLogger()

"""
Flows timeout, in seconds
"""
FLOW_TIMEOUT = 10

"""
Scheduling methods (used to choose a server to handle the service request)
"""
SCHED_RANDOM = 0
SCHED_ROUNDROBIN = 1
# SCHED_METHOD = SCHED_RANDOM
SCHED_METHOD = SCHED_ROUNDROBIN

"""
Statistics request period, in seconds
"""
STATS_REQ_PERIOD = 3

class Host (object):
    def __init__ (self, mac, ip, port):
        self.mac = mac
        self.ip = ip
        self.port = port
        self.req_n = 0

    def __str__ (self):
        return "MAC: " + str(self.mac) + " | IP: " + str(self.ip) + " | Port:" + str(self.port)

"""
Net params
"""
MAC_PREFIX = "00:00:00:00:00"
NET_PREFIX = "10.0.0"

"""
Switch params
Note that the VIRTUAL IP is f"{NET_PREFIX}.{SWITCH_IP_SUFFIX}"
"""
SWITCH_IP_SUFFIX = "13"
SWITCH_MAC_SUFFIX = "13"
SWITCH_HOST = Host(EthAddr(MAC_PREFIX + ":" + SWITCH_MAC_SUFFIX), IPAddr(NET_PREFIX + "." + SWITCH_IP_SUFFIX), None)

"""
Creates a list of hosts from 'start' to 'end'
"""
def fill_hosts_list (start, end):
    L = {}
    i = 0
    for host in range(start, end + 1):
        if host < 10:
            mac_suffix = "0" + str(host)
        else:
            mac_suffix = str(host)

        L[i] = Host(EthAddr(MAC_PREFIX + ":" + mac_suffix), IPAddr(NET_PREFIX + "." + str(host)), host)
        i += 1
    return L

"""
List of hosts
"""
CL_START = 1
CL_END = 6
SV_START = 7
SV_END = 12

CL_HOSTS = fill_hosts_list(CL_START, CL_END)
SV_HOSTS = fill_hosts_list(SV_START, SV_END)

"""
Gets a host by its mac
"""
def get_host_by_mac (hosts_list, mac):
    return next( (x for x in hosts_list.values() if str(x.mac) == str(mac)), None)

"""
Gets a host by its ip
"""
def get_host_by_ip (hosts_list, ip):
    return next( (x for x in hosts_list.values() if str(x.ip) == str(ip)), None)

class stats_req_thread (threading.Thread):
    def __init__ (self, connection, stop_flag):
        threading.Thread.__init__(self)

        self.connection = connection
        self.stop_flag = stop_flag

    """
    Periodically asks the statistics to the switch, till stop flag is raised
    """
    def run (self):
        while not self.stop_flag.wait(STATS_REQ_PERIOD):
            msg = of.ofp_stats_request()
            msg.type = of.OFPST_PORT
            msg.body = of.ofp_port_stats_request()
            self.connection.send(msg)

class proxy_load_balancer (object):
    def __init__ (self, connection):
        self.connection = connection

        # Timer should be global in order to be stopped when ConnectionDown event is raised
        global stop_flag
        stop_flag = threading.Event()
        stats_req_thread(self.connection, stop_flag).start()

        # If RR is the scheduling method, then choose a random server to start it
        if SCHED_METHOD is SCHED_ROUNDROBIN:
            self.last_server_idx = random.randint(0, len(SV_HOSTS))

        # Listen to the connection
        connection.addListeners(self)

    # AggregateFlowStats tables are deleted everytime a FlowMod is sent
    # Alternative is PortStats
    def _handle_PortStatsReceived (self, event):
        # log.info("Stats received: %s" % (str(flow_stats_to_list(event.stats))))
        pass

    def _handle_PacketIn (self, event):
        frame = event.parse()
        if frame.type == frame.ARP_TYPE:
            # log.info("Handling ARP Request from %s" % (frame.next.protosrc))
            self.arp_handler(frame, event)
        elif frame.type == frame.IP_TYPE:
            log.info("Handling Service request from %s" % (frame.next.srcip))
            self.service_handler(frame, event)

    """
    An ARP reply with switch fake MAC has to be sent
    """
    def arp_handler (self, frame, event):
        """
        TODO: Builds an ARP reply packet with the switch fake MAC
        """
        def build_arp_reply (frame, arp_request, is_client):

            if arp_request.opcode != arp.REQUEST:
                return
            arp_reply_msg = arp()
            arp_reply_msg.opcode = arp.REPLY
            arp_reply_msg.hwsrc = SWITCH_HOST.mac
            arp_reply_msg.hwdst = arp_request.hwsrc
            arp_reply_msg.protosrc = arp_request.protodst
            arp_reply_msg.protodst = arp_request.protosrc

            return arp_reply_msg


        arp_request = frame.next
        eth = ethernet(type=ethernet.ARP_TYPE)
        eth.src = SWITCH_HOST.mac
        eth.dst = arp_request.hwsrc
        eth.payload = build_arp_reply(frame, arp_request, True)
        packet_out = eth.pack()
        
        msg = of.ofp_packet_out()
        msg.data = packet_out
        action = of.ofp_action_output(port=event.port)
        msg.actions.append(action)

        # log.info("Sending OFP ARP Packet Out" % ())
        self.connection.send(msg)

    """
    Service packets should be balanced between all servers of the pool
    """
    def service_handler (self, frame, event):
        def choose_server ():
            if SCHED_METHOD is SCHED_RANDOM:
                chosen_server = random.choice(list(SV_HOSTS.values()))
            elif SCHED_METHOD is SCHED_ROUNDROBIN:
                if not hasattr(self, 'last_server_idx'):
                    self.last_server_idx = 0
                chosen_server = SV_HOSTS[self.last_server_idx]
                self.last_server_idx = (self.last_server_idx + 1) % len(SV_HOSTS)

            return chosen_server


        packet = frame.next
        chosen_server = choose_server()
        chosen_server.req_n += 1

        client_host = get_host_by_ip(CL_HOSTS, packet.srcip)

        msg = of.ofp_flow_mod()
        msg.idle_timeout = FLOW_TIMEOUT
        msg.match.dl_type = 0x0800
        msg.match.nw_src = chosen_server.ip
        msg.match.nw_dst = client_host.ip
        msg.match.dl_src = chosen_server.mac
        msg.match.dl_dst = SWITCH_HOST.mac
        msg.match.in_port = chosen_server.port
        msg.actions.append(of.ofp_action_nw_addr.set_src(SWITCH_HOST.ip))
        msg.actions.append(of.ofp_action_dl_addr.set_src(SWITCH_HOST.mac))
        msg.actions.append(of.ofp_action_dl_addr.set_dst(client_host.mac))
        msg.actions.append(of.ofp_action_output(port=client_host.port))

        self.connection.send(msg)

        msg = of.ofp_flow_mod()
        msg.idle_timeout = FLOW_TIMEOUT
        msg.data = event.ofp
        msg.buffer_id = event.ofp.buffer_id
        msg.match.dl_type = 0x0800
        msg.match.nw_src = client_host.ip
        msg.match.nw_dst = SWITCH_HOST.ip
        msg.match.dl_src = client_host.mac
        msg.match.dl_dst = SWITCH_HOST.mac
        msg.match.in_port = client_host.port
 
        msg.actions.append(of.ofp_action_nw_addr.set_dst(chosen_server.ip))
        msg.actions.append(of.ofp_action_dl_addr.set_src(SWITCH_HOST.mac))
        msg.actions.append(of.ofp_action_dl_addr.set_dst(chosen_server.mac))
        msg.actions.append(of.ofp_action_output(port=chosen_server.port))

        self.connection.send(msg)

        log.info("Chosen server for %s is %s" % (packet.srcip, chosen_server.ip))
        log.info("Sending OFP FlowMod Client -> Server path" % ())

"""
Controller
"""
class load_balancer (object):
    def __init__ (self):
        # Add listeners
        core.openflow.addListeners(self)

    """
    New connection from switch
    """
    def _handle_ConnectionUp (self, event):
        log.info("Switch connected" % ())
        # Create load balancer
        proxy_load_balancer(event.connection)

    """
    Connection from switch closed
    """
    def _handle_ConnectionDown (self, event):
        log.info("Switch disconnected" % ())
        # Stop stats req timer
        stop_flag.set()

def launch ():
    core.registerNew(load_balancer)
