#!/usr/bin/python

"""
Script created by VND - Visual Network Description (SDN version)
"""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch, IVSSwitch, UserSwitch
from mininet.link import Link, TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

net = Mininet( controller=RemoteController, link=TCLink, switch=OVSKernelSwitch )
c1 = net.addController( 'c1', controller=RemoteController )

class part1_topo(Topo):
    def build(self):
        # add hosts
        h1 = self.addHost( 'h1', mac='00:00:00:00:00:01', ip='10.0.0.1/8' )
        h2 = self.addHost( 'h2', mac='00:00:00:00:00:02', ip='10.0.0.2/8' )
        h3 = self.addHost( 'h3', mac='00:00:00:00:00:03', ip='10.0.0.3/8' )
        h4 = self.addHost( 'h4', mac='00:00:00:00:00:04', ip='10.0.0.4/8' )
        h5 = self.addHost( 'h5', mac='00:00:00:00:00:05', ip='10.0.0.5/8' )
        h6 = self.addHost( 'h6', mac='00:00:00:00:00:06', ip='10.0.0.6/8' )
        h7 = self.addHost( 'h7', mac='00:00:00:00:00:07', ip='10.0.0.7/8' )
        h8 = self.addHost( 'h8', mac='00:00:00:00:00:08', ip='10.0.0.8/8' )
        h9 = self.addHost( 'h9', mac='00:00:00:00:00:09', ip='10.0.0.9/8' )
        h10 = self.addHost( 'h10', mac='00:00:00:00:00:10', ip='10.0.0.10/8' )
        h11 = self.addHost( 'h11', mac='00:00:00:00:00:11', ip='10.0.0.11/8' )
        h12 = self.addHost( 'h12', mac='00:00:00:00:00:12', ip='10.0.0.12/8' )
    
        # add switches
        s1 = self.addSwitch( 's1', listenPort=6633, mac='00:00:00:00:00:13')
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)	
        self.addLink(h6, s1)
        self.addLink(h7, s1)
        self.addLink(h8, s1)
        self.addLink(h9, s1)
        self.addLink(h10, s1)
        self.addLink(h11, s1)
        self.addLink(h12, s1)


def configure():
    topo = part1_topo()
    net = Mininet(topo=topo, controller=RemoteController)
    net.start()

    CLI( net )

    net.stop()


if __name__ == '__main__':
    configure()

