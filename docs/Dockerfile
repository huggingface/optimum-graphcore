# Python 3.8 is needed because graphcore/pytorch:latest may be based on Ubuntu 18.04 with Python 3.6
# And the constraints transformers==4.20.0 is not compatible with Python 3.6
FROM graphcore/pytorch:3.1.0-ubuntu-20.04

ARG commit_sha
ARG clone_url
ARG clone_name

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg
RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash -
RUN apt-get -y install nodejs
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest

RUN git clone $clone_url  && cd $clone_name && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-graphcore[quality]