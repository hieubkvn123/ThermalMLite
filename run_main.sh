#!/bin/bash
cd $(pwd)
sudo python3 main.py -c vertical &
sleep 10 ; firefox --new-window http://localhost:5000 &
