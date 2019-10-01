# Make an intermediate image, just to get source code and compile cython code
FROM ubuntu:bionic as git_and_build
# install git and minimal dependencies to compile Cython modules
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  git \
  ssh \
  python-numpy \
  python-pip \
  python2.7-dev \
  build-essential

RUN python2 -c "import numpy; print numpy.__version__"
RUN pip install --no-cache-dir --compile --trusted-host pypi.python.org 'Cython==0.29.13'
RUN pip install --no-cache-dir --compile --trusted-host pypi.python.org 'setuptools==41.2.0'

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
# Pull source code
WORKDIR /
RUN pwd
RUN git clone git@bitbucket.org:marjens/cska.git

WORKDIR /cska
RUN git describe --always > git_commit
RUN python2 -c "import numpy; print numpy.__version__"
RUN python2 -m pip list
RUN python setup.py build_ext --inplace
RUN rm -rf /cska/build

## The final image. Start fresh
FROM ubuntu:bionic
RUN pwd
COPY --from=git_and_build /cska /cska
RUN apt-get update && apt-get install  --no-install-recommends -y \
    python-pip \
    python-numpy \
    python-scipy \
    python-matplotlib

WORKDIR /
RUN python2 -c "import numpy; print numpy.__version__"

RUN pwd
COPY run_req.txt /req.txt
RUN pip install --no-cache-dir --compile --trusted-host pypi.python.org "setuptools==41.2.0"
RUN pip install --no-cache-dir --compile --trusted-host pypi.python.org -r req.txt
ENV PYTHONPATH "/cska"
RUN mkdir target
WORKDIR /target
RUN rm -rf /var/lib/apt/lists
RUN rm -rf /usr/lib/gcc/x86_64-linux-gnu
# RUN apt-get purge ar gcc g++ perl manpages mime-support libpython-dev libpython2-dev
RUN apt-get purge gcc g++ perl manpages
RUN apt-get clean && apt-get autoremove
# ENTRYPOINT ["python", "/cska/bin/cska"]
CMD ["bash"]
