from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, FloatType
import time
import os
from pyspark.sql.functions import unix_timestamp, col, lit
from pyspark.ml import PipelineModel

# from influxdb_client import InfluxDBClient, Point, WriteOptions
# INFLUX_URL    = os.getenv("INFLUX_URL", "http://influxdb:8086")
# INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN", "my-token")
# INFLUX_ORG    = os.getenv("INFLUX_ORG", "my-org")
# INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "my-bucket")


# def write_batch_to_influx(batch_df, batch_id: int):
#     print("---write batch to influx-----")
#     batch_df.printSchema()
#     print("Cols:", batch_df.columns[:50])
#     if batch_df.isEmpty():
#         return
#     client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
#     write_api = client.write_api()
#     try:
#         for r in batch_df.collect():  
#             p = (
#                 Point("predictions")
#                 .tag("vehicle_type", r.vehicle_type)
#                 .field("prediction", float(r.prediction))
#                 .field("vehicle_speed", float(r.vehicle_speed))
#                 .field("vehicle_fuel", float(r.vehicle_fuel))
#                 .field("vehicle_CO2", float(r.vehicle_CO2))
#                 .time(r.ts)  # use aliased timestamp
#             )
#             write_api.write(bucket=INFLUX_BUCKET, record=p)
#     finally:
#         try: write_api.close()
#         finally:
#             try: client.close()
#             except: pass

if __name__ == "__main__":


    appName = "SparkStreamingApp"
    spark = SparkSession.builder \
        .appName(appName)\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")


    emission_topic = "emission_topic"
    model_path = 'hdfs://namenode3:9000/model'

    print("------------------------Spark Streaming----------------------------------")
    
    emission_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", emission_topic) \
        .load()
    
    emission_schema = StructType([
        StructField("timestep_time", FloatType()),
        StructField("vehicle_CO", FloatType()),
        StructField("vehicle_CO2", FloatType()),
        StructField("vehicle_HC", FloatType()),
        StructField("vehicle_NOx", FloatType()),
        StructField("vehicle_PMx", FloatType()),
        StructField("vehicle_angle", FloatType()),
        StructField("vehicle_eclass", StringType()),
        StructField("vehicle_electricity", FloatType()),
        StructField("vehicle_fuel", FloatType()),
        StructField("vehicle_id", StringType()),
        StructField("vehicle_lane", StringType()),
        StructField("vehicle_noise", FloatType()),
        StructField("vehicle_pos", FloatType()),
        StructField("vehicle_route", StringType()),
        StructField("vehicle_speed", FloatType()),
        StructField("vehicle_type", StringType()),
        StructField("vehicle_waiting", FloatType()),
        StructField("vehicle_x", FloatType()),
        StructField("vehicle_y", FloatType())
    ])

    emission_df_parsed = emission_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), emission_schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestep_time", (col("timestep_time") + unix_timestamp(lit("1970-01-01 00:00:00"))).cast("timestamp"))

    emission_df_parsed = emission_df_parsed.select(
        "vehicle_speed",
        "vehicle_type",
        "vehicle_fuel",
        "vehicle_CO2"
    )

    print("Loading model--------------------")
    model = PipelineModel.load(model_path)
    print(f"\nModel successfully loaded!\n")

    print("Pipeline stages:")
    for i, stage in enumerate(model.stages):
        print(f"Stage {i}: {stage.uid}, type={type(stage).__name__}")

    
    emission_df_predicted = model.transform(emission_df_parsed)
    emission_df_predicted = emission_df_predicted.select(
        "vehicle_speed",
        "vehicle_type",
        "vehicle_fuel",
        "vehicle_CO2",
        "prediction"
    )

    # query = emission_df_predicted.writeStream \
    # .outputMode("append") \
    # .format("console") \
    # .option("truncate", False) \
    # .start()

    predictions_path = 'hdfs://namenode3:9000/prediction'
    checkpoint_path = "hdfs://namenode3:9000/checkpoints/p3_pred"

    query = emission_df_predicted.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", predictions_path) \
    .option("checkpointLocation", checkpoint_path) \
    .option("truncate", False) \
    .start()

#------- Upis u influxdb ------------

    # query = (
    #     emission_df_predicted.writeStream
    #     .foreachBatch(write_batch_to_influx)
    #     .outputMode("append")
    #     .start()
    # )
    
    time.sleep(60)
    spark.streams.awaitAnyTermination()


