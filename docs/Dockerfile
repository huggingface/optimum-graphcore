FROM graphcore/pytorch:latest

# Need node14 to build doc HTML. Taken from https://askubuntu.com/a/1113339
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg
RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash -
RUN apt-get -y install nodejs
RUN npm install

# Python 3.8 is needed because graphcore/pytorch:latest may be based on Ubuntu 18.04 with Python 3.6
# And the constraints transformers==4.20.0 is not compatible with Python 3.6
RUN apt-get -y install python3.8 python3.8-dev python3.8-distutils

RUN git clone https://github.com/huggingface/optimum-graphcore.git
RUN python3.8 -m pip install --no-cache-dir --upgrade pip
RUN python3.8 -m pip install --no-cache-dir ./optimum-graphcore[quality]