# Project4 Part1: sdn_load_balancer

## Description

SDN Load Balancer. Example scheme with 6 clients <-> Switch (Transparent proxy, load balancer) <-> Pool of 6 servers.

The controller application (POX, Python) is connected to the switch in order to modify flow rules and balance the load among all servers. Clients aren't aware of backend servers, they only know about the transparent proxy (switch).

To run the SDN LB:
0) create a soft link of lb contorller: `ln -s ~/project-4/part1-loadbalancer/pox/* ~/pox/pox/misc/`

1) start pox via `sudo ~/pox/pox.py misc.load_balancer`

2) start mininet by: `sudo python3 ~/project-4/part1-loadbalancer/topos/part1.py`
