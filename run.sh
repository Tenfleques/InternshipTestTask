#!/bin/bash
con=$(docker container ls -a -f name=tf-local -q)
img=$(docker image ls | grep tf-local)
dir=$(pwd)

[[ ! -z  $con  ]] && docker start -ia tf-local &&  exit 0

[[ ! -z  $img  ]] && docker run -it -p 8888:8888 -p 6006:6008 -v $dir/human_segmentation:/tf/tendai --name tf-local tf-local:latest && exit 0

docker build --rm -f "Dockerfile" -t tf-local:latest .

docker run -it -p 8888:8888 -p 6006:6008 -v $dir/human_segmentation:/tf/tendai --name tf-local tf-local:latest