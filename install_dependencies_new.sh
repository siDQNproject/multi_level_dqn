#!/usr/bin/env bash

luarocks install cwrap
luarocks install paths
luarocks install nn
luarocks install cutorch
luarocks install cunn
luarocks install luafilesystem
luarocks install penlight
luarocks install sys
luarocks install xlua
luarocks install image
luarocks install env
luarocks install qtlua
luarocks install qttorch

echo "Installing Xitari ... "
cd /tmp
rm -rf xitari
git clone https://github.com/deepmind/xitari.git
cd xitari
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Xitari installation completed"

echo "Installing Alewrap ... "
cd /tmp
rm -rf alewrap
git clone https://github.com/deepmind/alewrap.git
cd alewrap
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Alewrap installation completed"



