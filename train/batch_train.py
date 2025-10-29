#PREDIKCIJA ZAGADJENJA CO2
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import os
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, isnan, when, count, mean, stddev, min, max
from pyspark.ml.evaluation import RegressionEvaluator


# spark = (
#     SparkSession.builder
#     .appName("BDS_P3")
#     .config("executor.memory", "4g")
#     .master("spark://spark-master3:7077")
#     .getOrCreate()
# )


spark = SparkSession.builder.appName("BDS_P3").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# print("Spark:", spark.version)
# sc = spark.sparkContext
# print("Hadoop:", sc._jvm.org.apache.hadoop.util.VersionInfo.getVersion())

#base_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.join(base_dir, "..", "data", "emission1.csv")
model_path = 'hdfs://namenode3:9000/model'

csv_path = "hdfs://namenode3:9000/data/emission1.csv"

df_emission = spark.read.csv(
  csv_path, 
  header=True,
  sep=";", 
  inferSchema=True
)

df_emission.cache() 

#Exploratory Data Analysis

print(f"Number of rows: {df_emission.count()}") #57347
print(f"Number of columns: {len(df_emission.columns)}") #20
df_emission.printSchema()
df_emission.show(10, truncate=False)

#df_emission.select("vehicle_CO2").describe().show()


#checking null value counts
df_null = df_emission.select([
    count(
        when(
            col(c).isNull() | isnan(c), c
        )
    ).alias(c)
    for c in df_emission.columns
])

#df_null.show()
df_emission = df_emission.dropna()


#describe important columns
columns_to_describe = [
    "vehicle_CO2", "vehicle_CO", "vehicle_NOx", "vehicle_HC", "vehicle_PMx",
    "vehicle_speed", "vehicle_fuel", "vehicle_waiting", "vehicle_pos"
]
for c in columns_to_describe:
    stats = (df_emission
             .agg(
                 min(col(c)).alias("min"),
                 max(col(c)).alias("max"),
                 mean(col(c)).alias("mean"),
                 stddev(col(c)).alias("stddev")
             )
             .collect()[0])

    print(f"\n=== {c} ===")
    print(f"min:   {stats['min']}")
    print(f"max:   {stats['max']}")
    print(f"mean:  {stats['mean']}")
    print(f"stddev:{stats['stddev']}")

#selecting features

df_emission = df_emission.select("vehicle_speed", "vehicle_type", "vehicle_fuel", "vehicle_CO2")

#converting to numerical feature
type_indexer = StringIndexer(inputCol="vehicle_type", outputCol="vehicle_type_idx")

assembler = VectorAssembler(
    inputCols=["vehicle_speed", "vehicle_fuel", "vehicle_type_idx"],
    outputCol="features"
)

regressor = GBTRegressor(
    labelCol="vehicle_CO2",
    featuresCol="features",
    maxIter=50
)

pipeline = Pipeline(stages=[type_indexer, assembler, regressor])

train_data, test_data = df_emission.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)

predictions = model.transform(test_data)

evaluator_rmse = RegressionEvaluator(
    labelCol="vehicle_CO2", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)
print(f"Test RMSE: {rmse:.2f}")

evaluator_r2 = RegressionEvaluator(
    labelCol="vehicle_CO2", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print(f"Test R2: {r2:.4f}")

try:
    model.write().overwrite().save(model_path)
except Exception as e:
    print(e)