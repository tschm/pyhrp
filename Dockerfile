FROM python:3.8.7-buster as builder

# File Author / Maintainer
MAINTAINER Thomas Schmelzer "thomas.schmelzer@gmail.com"

COPY . /tmp/pyhrp

RUN pip install --no-cache-dir -r /tmp/pyhrp/requirements.txt && \
    pip install --no-cache-dir /tmp/pyhrp && \
    rm -r /tmp/pyhrp


# ----------------------------------------------------------------------------------------------------------------------
FROM builder as test

# COPY tools needed for testing into the image
RUN pip install --no-cache-dir  pytest pytest-cov pytest-html

# Install package by Robert Martin
# RUN pip install --no-cache-dir PyPortfolioOpt

# COPY the tests over
COPY test /pyhrp/test

CMD py.test --cov=pyhrp  --cov-report html:artifacts/html-coverage --cov-report term --html=artifacts/html-report/report.html /pyhrp/test

# ----------------------------------------------------------------------------------------------------------------------
FROM builder as lint

RUN pip install --no-cache-dir pylint

WORKDIR /pyhrp

CMD pylint pyhrp
