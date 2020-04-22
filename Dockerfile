FROM python:3.7.7-slim-stretch as builder

# File Author / Maintainer
MAINTAINER Thomas Schmelzer "thomas.schmelzer@gmail.com"

# this will be user root regardless whether home/beakerx is not
COPY . /tmp/pyhrp

RUN buildDeps='gcc g++' && \
    apt-get update && apt-get install -y $buildDeps --no-install-recommends && \
    pip install --no-cache-dir pandas==0.25.3 requests==2.22.0 && \
    pip install --no-cache-dir /tmp/pyhrp && \
    rm -r /tmp/pyhrp && \
    apt-get purge -y --auto-remove $buildDeps



# ----------------------------------------------------------------------------------------------------------------------
FROM builder as test

# COPY tools needed for testing into the image
RUN pip install --no-cache-dir  httpretty pytest pytest-cov pytest-html sphinx requests-mock mock

# COPY the tests over
COPY test /pyhrp/test

CMD py.test --cov=pyhrp  --cov-report html:artifacts/html-coverage --cov-report term --html=artifacts/html-report/report.html /pyhrp/test
