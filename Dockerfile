FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel


RUN apt-get update && apt-get install -y locales software-properties-common openssh-client git wget virtualenv

RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

RUN python --version
RUN pip list

# ADD requirements_pre.txt .
# RUN pip install --user -r requirements_pre.txt

ENV SRCDIR=pytorch-3dunet
ADD $SRCDIR/requirements.txt $SRCDIR/requirements.txt
RUN cd $SRCDIR && pip install --user -r requirements.txt

# ADD $SRCDIR $SRCDIR
CMD cd $SRCDIR && python train.py --config resources/train_config_our_docker.yaml
