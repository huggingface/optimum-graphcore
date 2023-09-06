FROM graphcore/pytorch:3.2.0-ubuntu-20.04

ARG commit_sha
ARG clone_url
ARG clone_name

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg
RUN curl -sL https://deb.nodesource.com/setup_18.x  | bash -
RUN apt-get -y install nodejs
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest

RUN git clone $clone_url  && cd $clone_name && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN cd $clone_name && python3 -m pip install --no-cache-dir .[quality]
