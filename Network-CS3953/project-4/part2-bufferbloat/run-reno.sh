#!/usr/bin/env bash

# Note: Mininet must be run as root.  So invoke this shell script
# using sudo.

time=90
bwnet=1.5
# TODO: If you want the RTT to be 20ms what should the delay on each
# link be?  Set this value correctly.
delay=5

iperf_port=5001

for qsize in 20 100; do
    dir=reno-q$qsize

    # TODO: Run bufferbloat.py here...
    python3 bufferbloat.py --dir $dir --time $time --bw-net $bwnet --delay $delay --maxq $qsize

    # TODO: Ensure the input file names match the ones you use in
    # bufferbloat.py script.  Also ensure the plot file names match
    # the required naming convention when submitting your tarball.
    python3 plot_queue.py -f $dir/q.txt -o png/reno-buffer-q$qsize.png
    python3 plot_ping.py -f $dir/ping.txt -o png/reno-rtt-q$qsize.png
done
