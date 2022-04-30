FROM wnsq0317/pytorch:1.11-cuda11.3.1

LABEL maintainer="wnsq0317@naver.com" purpose="study"

RUN apt-get update
RUN apt-get install git -y

ADD requirement.txt /tmp

WORKDIR /var

RUN pip install -r requirement.txt

WORKDIR /

EXPOSE 32145