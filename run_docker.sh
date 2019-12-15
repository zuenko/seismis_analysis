IMG=3dnet-pytorch
# docker build -t $IMG -f Dockerfile .
# nvidia-docker run -it -v $(readlink -f our.h5):/cdir/our.h5 $IMG bash -c "cd \$SRCDIR && python train.py --config resources/train_docker_small.yaml"
# nvidia-docker run -it -v $(pwd):/cdir -v $(readlink -f our.h5):/cdir/our.h5 $IMG bash
nvidia-docker run -it -v $(pwd):/cdir -v $(readlink -f our.h5):/cdir/our.h5 $IMG bash  -c "cd /cdir/pytorch-3dunet && python predict.py --config resources/test_docker.yaml"
