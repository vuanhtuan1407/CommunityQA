FROM tensorflow/tensorflow:1.12.3-gpu-py3
#FROM tensorflow/tensorflow:1.12.3-py3
RUN pip install bert-serving-server
RUN #pip install -U bert-serving-server[http]
RUN cat /etc/lsb-release
RUN apt update
RUN apt install wget
RUN mkdir /app
#VOLUME /app/data
#RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#RUN wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
#RUN unzip uncased_L-24_H-1024_A-16.zip
RUN unzip uncased_L-12_H-768_A-12.zip
#RUN unzip chinese_L-12_H-768_A-12.zip
#RUN mv uncased_L-24_H-1024_A-16 /app
RUN mv uncased_L-12_H-768_A-12 /app
#RUN mv chinese_L-12_H-768_A-12 /app
RUN wget https://github.com/datar-ai/bert-as-service/raw/master/docker/bert-serving-start.sh
RUN mv ./bert-serving-start.sh /app
RUN chmod 777 /app/bert-serving-start.sh
WORKDIR /app
ENTRYPOINT ["/usr/local/bin/bert-serving-start -num_worker=1 -model_dir /app/uncased_L-12_H-768_A-12"]
CMD [""]
