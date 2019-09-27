# Intermediate, just to pull source code!
FROM ubuntu as intermediate

# install git
RUN apt-get update
RUN apt-get install -y git

# add credentials on build
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN chmod 700 /root/.ssh
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa

# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts

RUN git clone git@bitbucket.org:marjens/cska.git

# Here's the actual build of the image. Does not expose my SSH_KEY!!!

# Use an official Python runtime as a parent image
# FROM  madnight/docker-alpine-wkhtmltopdf
# RUN apk update && apk add py2-pip

FROM python:2.7-slim
RUN pwd
RUN apt-get update && apt-get -y install libblas3 liblapack3 build-essential zlib1g-dev # wkhtmltopdf
COPY --from=intermediate /cska /cska

WORKDIR /
COPY requirements.txt /req.txt
RUN python --version
RUN python2 --version
RUN pip install --trusted-host pypi.python.org "numpy==1.16.5"
RUN pwd
RUN pip install --trusted-host pypi.python.org -r req.txt
ENV PYTHONPATH "/cska"
WORKDIR /cska
RUN python setup.py build_ext --inplace
ENTRYPOINT ["python", "/cska/bin/cska"]
