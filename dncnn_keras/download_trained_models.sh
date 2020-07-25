#!/bin/bash
cd trained_models/

fileid="1Z33zkZkeDG4g94UHpid_u1RvXFiLGAzF"
filename="val_several_classes_trained_models.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip val_several_classes_trained_models.zip

rm -rf val_several_classes_trained_models.zip
rm -rf cookie

cd ..