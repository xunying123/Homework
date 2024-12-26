# Project4 Part2: Mininet Bufferbloat

This directory contains starter and support code for project 4 part2. You may want to
version control this file for your team and will need to update this readme file
with your own information in your final submission.

# Submission Details

Name:吴硕
StudentID: 522030910094

## Instructions to reproduce your results:
  > sudo ./run-bbr.sh
  >
  > sudo ./run-reno.sh

## Answers to the questions:

### Step 2
  1. > Q20：平均: 0.4201433999999999 seconds 标准差：0.16058901663416733
     >
     > Q100：平均：2.2014047037037034 标准差：1.0528676996486985
  2. > 较小的缓冲区可以更快的发现网络中的堵塞并及时控制，因此可以将延迟控制在一个相对较低的水平。较大的缓冲区对网络的堵塞更不敏感，网络中的数据流控制滞后，延迟更大，且数据在缓存中停留的时间更长。因此应用程序层面的感受就是过大的延迟。
  3. > 最大队列长度是1000，最长等待时间为120ms
  4. > 对于Q20，RTT随时间增加而周期性变化，只有在最开始时可以达到20ms，但随后会在150-300ms之间来回震荡。对于Q100，RTT先经过短时间的震荡，随后随着时间不断增加（由于拥塞控制）
  5. > 方法一：减小缓冲区的大小，从而减少数据包在缓冲区的停留时间，使得网络更快的感知到拥塞
     >
     > 方法二：主动的丢包，当队列过大的时候选择性的丢掉部分包

### Step 3
  1. > Q20：平均: 0.25890658333333333 seconds 标准差：0.056748247556948586
     >
     > Q100：平均：0.25507125 seconds标准差：0.05797530036414214
  2. > Q100获取时间更短一些，但二者差别不大。因为无论是Q100还是Q20，二者的buffer都没有满，也就都没有明显的缓冲区膨胀问题，因此时间应当差不多。
  3. > bbr的buffer均控制在13以下，虽然也是震荡，但一直是动态调整，从来没有满；而reno的buffer不仅会震荡，还会满导致大量丢包。rtt同理，bbr一直维持在一个较低的值附近；reno不仅震荡。且方差较大，还会爆buffer触发大量丢包
  4. > 不完全，现有的bbr算法是依靠对网络状态的一个估计，但不可能适用于所有状态下的网络。对于一些新的网络架构，或者更加复杂的网络状况，现有的解决算法可能就完全失去作用。
