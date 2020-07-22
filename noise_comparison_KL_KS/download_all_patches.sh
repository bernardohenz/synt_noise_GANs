#!/bin/bash
cd all_patches/

fileid=""
filename="data_per_ISO_lighting_camera.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip data_per_ISO_lighting_camera.zip

rm -rf data_per_ISO_lighting_camera.zip
rm -rf cookie

cd ..