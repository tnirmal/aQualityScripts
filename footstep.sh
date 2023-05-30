#!/bin/bash
cd /home/algoiei/Downloads/download_2022-11-11_17-55-49/IEI_Intelligent_System_Management_Module_Plus/LINUX_V/iei_ismm_plus_SDK_V1.5/iei_ismm_plus
sudo cp lib/x64/libismmplus.so /usr/lib
cd module/
make
sudo insmod acpi_call.ko
cd ..
cd sample/
make

sudo python3 footstep.py
