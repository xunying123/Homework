from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController
from mininet.log import info


class part1_topo(Topo):
    def build(self):
        s1 = self.addSwitch("s1")
        h1 = self.addHost("h1", mac="00:00:00:00:00:01", ip="10.0.1.2/24")
        h2 = self.addHost("h2", mac="00:00:00:00:00:02", ip="10.0.0.2/24")
        h3 = self.addHost("h3", mac="00:00:00:00:00:03", ip="10.0.0.3/24")
        h4 = self.addHost("h4", mac="00:00:00:00:00:04", ip="10.0.1.3/24")
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)


topos = {"part1": part1_topo}


def configure():
    setLogLevel( 'info' )
    topo = part1_topo()
    net = Mininet(topo=topo, controller=RemoteController)
    net.start()
    
    info("Testing connectivity with pingAll\n")
    net.pingAll()

    h1, h4 = net.get('h1'), net.get('h4')

    info("Starting iperf server on h1\n")

    h1.cmd('iperf -s -p 5001 &')

    info("Testing bandwidth from h4 to h1\n")

    result = h4.cmd('iperf -c ' + h1.IP() + ' -p 5001 -t 5')

    info("Test Results:\n" + result)

    s1 = net.get('s1')

    info("Dumping flow table of switch s1\n")

    flows = s1.dpctl('dump-flows')
    
    info(flows)

    net.stop()


if __name__ == "__main__":
    configure()
