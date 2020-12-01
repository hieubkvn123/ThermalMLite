#!/bin/bash
cd /home/lattepanda/ThermalX-custom-1
#sudo chmod 0777 /dev/ttyACM0
#sudo chmod 0777 /dev/ttyACM1
sudo python3 main.py &
sleep 10 ; firefox -url http://localhost:5000 &
# sleep 2 ; xdotool search --sync --onlyvisible --class "Firefox" windowactivate key F11
