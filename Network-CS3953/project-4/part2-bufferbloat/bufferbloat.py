from mininet.topo import Topo
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.log import lg, info
from mininet.util import dumpNodeConnections
from mininet.cli import CLI

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
from argparse import ArgumentParser

from monitor import monitor_qlen

import sys
import os
import math

parser = ArgumentParser(description="Bufferbloat tests")
parser.add_argument(
    "--bw-host", "-B", type=float, help="Bandwidth of host links (Mb/s)", default=1000
)

parser.add_argument(
    "--bw-net",
    "-b",
    type=float,
    help="Bandwidth of bottleneck (network) link (Mb/s)",
    required=True,
)

parser.add_argument(
    "--delay", type=float, help="Link propagation delay (ms)", required=True
)

parser.add_argument("--dir", "-d", help="Directory to store outputs", required=True)

parser.add_argument(
    "--time", "-t", help="Duration (sec) to run the experiment", type=int, default=10
)

parser.add_argument(
    "--maxq",
    type=int,
    help="Max buffer size of network interface in packets",
    default=100,
)

# Linux uses CUBIC-TCP by default that doesn't have the usual sawtooth
# behaviour.  For those who are curious, invoke this script with
# --cong cubic and see what happens...
# sysctl -a | grep cong should list some interesting parameters.
parser.add_argument(
    "--cong", help="Congestion control algorithm to use", default="reno"
)

# Expt parameters
args = parser.parse_args()

class BBTopo(Topo):
    "Simple topology for bufferbloat experiment."

    def build(self, n=2):
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")

        switch = self.addSwitch("s0")

        self.addLink(h1, switch, bw=args.bw_host, delay=f"{args.delay}ms", max_queue_size=args.maxq)
        self.addLink(switch, h2, bw=args.bw_net, delay=f"{args.delay}ms", max_queue_size=args.maxq)

# Simple wrappers around monitoring utilities.  You are welcome to
# contribute neatly written (using classes) monitoring scripts for
# Mininet!

def start_iperf(net):
    h1 = net.get("h1")
    h2 = net.get("h2")
    print("Starting iperf server...")
    
    server = h2.popen("iperf -s -w 16m")

    client = h1.popen(f"iperf -c {h2.IP()} -t {args.time + 3} -w 16m")

    return server, client

def start_qmon(iface, interval_sec=0.1, outfile="q.txt"):
    monitor = Process(target=monitor_qlen, args=(iface, interval_sec, outfile))
    monitor.start()
    return monitor

def start_ping(net):
    h1 = net.get("h1")
    h2 = net.get("h2")
    ping_cmd = f"ping {h2.IP()} -i 0.1 > {args.dir}/ping.txt"
    ping_proc = h1.popen(ping_cmd, shell=True)
    return ping_proc

def start_webserver(net):
    h1 = net.get("h1")
    proc = h1.popen("python3 webserver.py", shell=True)
    sleep(1)
    return [proc]

def measure_webpage_transfer(net):
    h1 = net.get("h1")
    h2 = net.get("h2")
    webserver_ip = h1.IP()
    curl_cmd = f"curl -o /dev/null -s -w '%{{time_total}}' http://{webserver_ip}/"
    total_time = []
    for _ in range(3):
        h2_curl = h2.popen(curl_cmd, shell=True)
        c_time = float(h2_curl.communicate()[0].decode('utf-8').strip())
        total_time.append(c_time)

    return total_time

def bufferbloat():
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    if not os.path.exists('./png'):
        os.makedirs('./png')
    os.system("sysctl -w net.ipv4.tcp_congestion_control=%s" % args.cong)
    topo = BBTopo()
    net = Mininet(topo=topo, link=TCLink)
    # net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
    net.start()

    dumpNodeConnections(net.hosts)

    net.pingAll()

    qmon = start_qmon(iface="s0-eth2", outfile="%s/q.txt" % (args.dir))

    start_ping(net)

    start_iperf(net)

    start_webserver(net)

    start_time = time()
    total_time = []
    while True:
        sleep(5)
        now = time()
        timerr = measure_webpage_transfer(net)
        total_time.extend(timerr)
        delta = now - start_time
        if delta > args.time:
            break
        print("%.1fs left..." % (args.time - delta))

    avg_time = sum(total_time) / len(total_time)
    variance = sum((x - avg_time) ** 2 for x in total_time) / len(total_time)
    std_dev = math.sqrt(variance)

    with open(f"{args.dir}/curl_times.txt", "a") as f:
        f.write(f"Average download time: {avg_time} seconds and standard deviation {std_dev}\n")

    # CLI(net)

    qmon.terminate()
    net.stop()
    # Ensure that all processes you create within Mininet are killed.
    # Sometimes they require manual killing.
    Popen("pgrep -f webserver.py | xargs kill -9", shell=True).wait()

if __name__ == "__main__":
    bufferbloat()
