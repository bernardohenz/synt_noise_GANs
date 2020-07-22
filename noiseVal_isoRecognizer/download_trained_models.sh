#!/bin/bash
cd trained_models/

fileid="1afl-UI-8hpSypCtbsw8zyp1eXdmAC77w"
filename="trained_models.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip trained_models.zip

rm -rf trained_models.zip
rm -rf cookie

cd ..