# Python 3.8 is needed because graphcore/pytorch:latest may be based on Ubuntu 18.04 with Python 3.6
# And the constraints transformers==4.20.0 is not compatible with Python 3.6
FROM graphcore/pytorch:3.0.0-ubuntu-20.04

ARG commit_sha
ARG clone_url

# Need node14 to build doc HTML. Taken from https://askubuntu.com/a/1113339
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg
RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash -
RUN apt-get -y install nodejs
RUN npm install

RUN git clone $clone_url  && cd optimum-graphcore && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-graphcore[quality]