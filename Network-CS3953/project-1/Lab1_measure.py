#!/usr/bin/env python

import os
import time
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.node import OVSController
from Lab1_topo import AssignmentNetworks

if __name__ == '__main__':
       setLogLevel( 'info' )

       # Create data network
       topo = AssignmentNetworks()
       net = Mininet(topo=topo, controller=OVSController, link=TCLink, autoSetMacs=True,
                     autoStaticArp=True)

       # Run network
       net.start()

       # Fill in start

       # example 1: path latency measurement
       # get host
       h1 = net.get('h1')
       # cmd. Redirect the output of the `ping` cmd to [res_file], and '&' means to run this cmd in the background
       res_filea = 'latency_h1-h4.txt'
       h1.cmd('ping -c 20 10.0.0.4 >> {} &'.format(res_filea))

       h5 = net.get('h5')
       res_fileb = 'latency_h5-h6.txt'
       h5.cmd('ping -c 20 10.0.0.6 >> {} &'.format(res_fileb))

       time.sleep(50)

       # example 2: path throughput measurement
       # get host
       h1, h4 = net.get('h1'), net.get('h4')
       # cmd
       res_file1 = 'throughput_h1-h4.txt'
       res_file4 = 'throughput_h1-h4.txt'
       h1.cmd('python3 Iperfer.py -s -p 8080 >> {} &'.format(res_file1))
       h4.cmd('python3 Iperfer.py -c -h 10.0.0.1 -p 8080 -t 20 >> {} &'.format(res_file4))

       h5, h6 = net.get('h5'), net.get('h6')
       # cmd
       res_file5 = 'throughput_h5-h6.txt'
       res_file6 = 'throughput_h5-h6.txt'
       h5.cmd('python3 Iperfer.py -s -p 8088 >> {} &'.format(res_file5))
       h6.cmd('python3 Iperfer.py -c -h 10.0.0.5 -p 8088 -t 20 >> {} &'.format(res_file6))

       time.sleep(100)
       # Fill in end
       net.stop()