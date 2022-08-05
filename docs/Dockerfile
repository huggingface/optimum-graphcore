FROM graphcore/pytorch:latest

ARG commit_sha

# Need node14 to build doc HTML. Taken from https://askubuntu.com/a/1113339
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg
RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash -
RUN apt-get -y install nodejs
RUN npm install

RUN git clone https://github.com/huggingface/optimum-graphcore.git  && cd optimum-graphcore && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-graphcore[quality]