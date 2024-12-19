from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController
from mininet.log import info


class part2_topo(Topo):
    def build(self):
        # add switches
        s1 = self.addSwitch("s1")
        s2 = self.addSwitch("s2")
        s3 = self.addSwitch("s3")
        cores21 = self.addSwitch("cores21")
        dcs31 = self.addSwitch("dcs31")
        # add hosts
        h10 = self.addHost(
            "h10",
            mac="00:00:00:00:00:01",
            ip="10.0.1.10/24",
            defaultRoute="via 10.0.1.1",
        )
        h20 = self.addHost(
            "h20",
            mac="00:00:00:00:00:02",
            ip="10.0.2.20/24",
            defaultRoute="via 10.0.2.1",
        )
        h30 = self.addHost(
            "h30",
            mac="00:00:00:00:00:03",
            ip="10.0.3.30/24",
            defaultRoute="via 10.0.3.1",
        )
        serv1 = self.addHost(
            "serv1",
            mac="00:00:00:00:00:04",
            ip="10.0.4.10/24",
            defaultRoute="via 10.0.4.1",
        )
        hnotrust1 = self.addHost(
            "hnotrust1",
            mac="00:00:00:00:00:05",
            ip="172.16.10.100/24",
            defaultRoute="via 172.16.10.1",
        )
        # add links
        self.addLink(h10, s1)
        self.addLink(h20, s2)
        self.addLink(h30, s3)
        self.addLink(s1, cores21)
        self.addLink(s2, cores21)
        self.addLink(s3, cores21)
        self.addLink(serv1, dcs31)
        self.addLink(cores21, dcs31)
        self.addLink(hnotrust1, cores21)

def configure():
    setLogLevel( 'info' )
    topo = part2_topo()
    net = Mininet(topo=topo, controller=RemoteController)
    net.start()

    info("Testing connectivity with pingAll\n")
    net.pingAll()

    info("Testing connectivity with pingAll\n")
    net.pingAll()

    hnotrust1, h10, serv1 = net.get('hnotrust1'), net.get('h10'), net.get('serv1')

    info("Starting iperf server on hnotrust1\n")
    hnotrust1.cmd('iperf -s -p 5001 &')
    info("Testing bandwidth from h10 to hnotrust1\n")
    result = h10.cmd('iperf -c ' + hnotrust1.IP() + ' -p 5001 -t 5')
    info("Test Results:\n" + result)

    info("Starting iperf server on serv1\n")
    serv1.cmd('iperf -s -p 5002 &')
    info("Testing bandwidth from h10 to serv1\n")
    result = h10.cmd('iperf -c ' + serv1.IP() + ' -p 5002 -t 5')
    info("Test Results:\n" + result)

    s1 = net.get('s1')
    info("Dumping flow table of switch s1\n")
    flows = s1.dpctl('dump-flows')
    info(flows)

    s2 = net.get('s2')
    info("Dumping flow table of switch s2\n")
    flows = s2.dpctl('dump-flows')
    info(flows)

    s3 = net.get('s3')
    info("Dumping flow table of switch s3\n")
    flows = s3.dpctl('dump-flows')
    info(flows)

    cores21 = net.get('cores21')
    info("Dumping flow table of switch cores21\n")
    flows = cores21.dpctl('dump-flows')
    info(flows)

    dcs31 = net.get('dcs31')
    info("Dumping flow table of switch dcs31\n")
    flows = dcs31.dpctl('dump-flows')
    info(flows)
    net.stop()


if __name__ == "__main__":
    configure()
