FROM bitnami/spark:latest

WORKDIR /app
COPY spark_streaming.py .
RUN cd /app && pip install numpy==1.26.4
#influxdb-client

CMD ["/opt/bitnami/spark/bin/spark-submit", "--master", "spark://spark-master3:7077", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0", "spark_streaming.py"]