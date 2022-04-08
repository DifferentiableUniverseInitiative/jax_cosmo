FROM tensorflow/tensorflow:2.4.0-gpu
# This contains CUDA 11.0 and CUDNN 8.X.X
# Could try using later versions

# Basic dependencies
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get update && apt-get install -y wget pkg-config curl \
        git cmake make swig && apt-get clean all
RUN apt-get update && apt-get install -y libcfitsio-dev libfftw3-dev \
        libgsl-dev libbz2-dev  && apt-get clean all

# Install CCL manually
RUN cd /opt \
    && git clone https://github.com/LSSTDESC/CCL.git \
    && cd CCL \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd .. \
    && python setup.py install \
    && rm -rf /opt/CCL

# Install pytest for unit-testing
RUN pip install --upgrade pip
RUN pip install --no-cache-dir pytest setuptools astropy fitsio \
        scipy matplotlib jupyter
RUN pip install --no-cache-dir jax==0.2.10
RUN pip install --no-cache-dir jaxlib==0.1.62+cuda110 \
                                             -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /opt/jax_cosmo
COPY . .

CMD ["/bin/bash", "-l"]
