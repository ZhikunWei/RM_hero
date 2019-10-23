#!/bin/sh
echo dji | sudo -S /home/dji/jetson_clocks.sh
cd /home/dji/Desktop/hero2.3/
echo dji | sudo -S /home/dji/Desktop/hero2.3/auto_start.sh && sleep 10m
