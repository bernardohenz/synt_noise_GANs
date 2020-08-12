#!/bin/bash
cd checkpoints/

fileid="1x_ONMjQEGxKU-F1zfbv1qluhYyOLa0z4"
filename="checkpoints.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip checkpoints.zip

rm -rf checkpoints.zip
rm -rf cookie

cd ..