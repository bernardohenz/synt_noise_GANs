#!/bin/bash
cd sample_imgs/

fileid="17_EErt51NOrOhS15fRID3R9HVM2F6lUH"
filename="sample_imgs.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip sample_imgs.zip

rm -rf sample_imgs.zip
rm -rf cookie

cd ..