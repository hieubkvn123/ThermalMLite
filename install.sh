### Install Python and python git ###
case_orientation=$1
sudo apt install python3-pip
sudo apt install python3 python3-dev python3-pip

### Installing opencv ###
sudo apt install python3-opencv

### install xdotool and seekware libraries ###
sudo apt install xdotool
sudo apt install cmake
sudo cp include/seekware.h /usr/include/seekware.h
sudo cp lib/libseekware.so.3.4 /usr/lib/libseekware.so.3.4

### Python dependencies ###
sudo python3 -m pip install flask
sudo python3 -m pip install cvlib
sudo python3 -m pip install opencv-python==3.4.8.29 # or 3.4.6.27

# installing tensorflow will automatically install scipy, numpy
sudo python3 -m pip install tensorflow==2.1 --verbose # for ThermalMLite 2.1, ThermalM 1.14
sudo python3 -m pip install numpy
sudo python3 -m pip install Pillow
sudo python3 -m pip install matplotlib==3.2.1
sudo python3 -m pip install imutils
sudo python3 -m pip install dlib --user --verbose
sudo python3 -m pip install face_recognition
sudo python3 -m pip install onnx
sudo python3 -m pip install onnxruntime
sudo python3 -m pip install pandas
sudo python3 -m pip install playsound
sudo python3 -m pip install scikit-learn
sudo python3 -m pip install requests

### Others ###
sudo mkdir static/img/incidents
echo "QT_X11_NO_MITSHM=1" | sudo tee -a /etc/environment
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee -a /etc/sudoers

### Setting background image ###
gsettings set org.gnome.desktop.background picture-options "scaled"
mv static/img/thermalm.jpeg ~/Pictures/thermalm.jpeg ### move bg image to home dir ###
gsettings set org.gnome.desktop.background picture-uri "file:///home/lattepanda/Pictures/thermalm.jpeg"

### create the incidents folder ###
sudo mkdir static/img/incidents
sudo chmod 0777 static/img/incidents/*.jpeg
sudo chmod 0777 ./static/img/incidents

### Change mode of audio files ###
sudo chmod 0777 ./audio
sudo chmod 0777 audio/*.mp3

### create a startup file ###
sudo rm run_main.sh ### remove old version ###
touch run_main.sh
echo "!/bin/bash" | sudo tee -a run_main.sh
echo "cd $(pwd)"|sudo tee -a run_main.sh
if [ $case_orientation = "vertical" ];
then
	echo "sudo python3 main.py -c vertical &" | sudo tee -a run_main.sh
else	
	echo "sudo python3 main.py -c horizontal &" | sudo tee -a run_main.sh
fi

echo "sleep 10 ; firefox --new-window http://localhost:5000 &" | sudo tee -a run_main.sh
sudo chmod +x run_main.sh
sudo cp run_main.sh ~/run_main.sh
