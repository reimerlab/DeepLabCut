FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Satoshi Murashige

RUN apt-get -y update --fix-missing && apt-get -y upgrade
RUN apt-get install -y sudo wget bzip2 git vim cmake xserver-xorg-dev libgl1-mesa-dev unzip && \
    apt-get install -y x11-apps

# settings
ARG user_name=ubuntu
ARG user_id=1000
ARG group_name=ubuntu
ARG group_id=1000

# create user
RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -d /home/${user_name} \
    --create-home --shell /bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}

# user settings
ENV HOME /home/${user_name}
ENV LANG en_US.UTF-8

# Intall Anaconda
RUN echo 'export PATH=$HOME/anaconda3/bin:$PATH' > /etc/profile.d/anaconda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O $HOME/anaconda.sh && \
    /bin/bash $HOME/anaconda.sh -b -p $HOME/anaconda3 && \
    rm $HOME/anaconda.sh

ENV PATH $HOME/anaconda3/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN chown -R ${user_name}:${group_name} $HOME/anaconda3

##### Install Deeplabcut and its dependencies #####

USER ${user_name}
WORKDIR /work

# Install DeepLabCut
RUN conda install -y tensorflow==1.8 tensorflow-gpu==1.8
RUN pip install --upgrade --force-reinstall pip==9.0.3
RUN pip install deeplabcut==2.0.5
RUN pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl

# Install dependencies of wxPython
RUN sudo apt-get install -y libgtk2.0-dev libgtk-3-dev \
    libjpeg-dev libtiff-dev \
    libsdl1.2-dev libgstreamer-plugins-base1.0-dev \
    libnotify-dev freeglut3 freeglut3-dev libsm-dev \
    libwebkitgtk-dev libwebkitgtk-3.0-dev

# Install plotly and bokeh for plotting interactive plotting in jupyter
RUN pip install plotly 
RUN pip install bokeh==1.0.4
RUN pip install colorcet==1.0.1

# install seaborn and pandas
RUN pip install seaborn
RUN pip install pandas

# Install datajoint
RUN pip install datajoint
# need to install some other package for dj.ERD to work (graphviz-dev?)


# Install ffmpeg
RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository ppa:jonathonf/ffmpeg-4
RUN sudo apt-get update
RUN sudo apt-get install -y ffmpeg

# add ownership
RUN sudo chown -R ${user_name}:${group_name} /work

# Install ImageMagick
RUN sudo apt-get update -y
RUN sudo apt-get install -y curl tar file xz-utils build-essential
RUN wget https://imagemagick.org/download/ImageMagick.tar.gz

WORKDIR /work/ImageMagick
RUN sudo chown -R ${user_name}:${group_name} /work/ImageMagick

WORKDIR /work

RUN tar -xzf ImageMagick.tar.gz -C ImageMagick --strip-components 1
WORKDIR /work/ImageMagick

RUN ./configure --prefix /usr/local
RUN make
RUN sudo make install
RUN sudo ldconfig /usr/local/lib

WORKDIR /work
CMD ["/bin/bash"]

