#!/bin/bash
cd checkpoints/

fileid="1INIqDRjVP1n0fvz8T8F55IvGc1znXv_G"
filename="checkpoints.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip checkpoints.zip

rm -rf checkpoints.zip
rm -rf cookie

cd ..