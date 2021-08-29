from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np
import getpass
import sys
import ast


def main(spark, netID, frac):
    sc = spark.sparkContext
    train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet').sample(frac)
    train = train.toDF('user_id','count','track_id', '__index_col_0__').drop('__index_col_0__').repartition(20)
    train.createOrReplaceTempView('train')
    val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet').toDF('user_id','count','track_id','__index_col_0__').drop('__index_col_0__').repartition(20)
    val.createOrReplaceTempView('val')
    test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet').toDF('user_id','count','track_id','__index_col_0__').drop('__index_col_0__').repartition(20)
    test.createOrReplaceTempView('test')
    #train.show()
    
    # Convert string columns to int columns for use with ALS
    indexers = [StringIndexer(inputCol='user_id', outputCol='user_int', handleInvalid='keep'),
                StringIndexer(inputCol='track_id', outputCol='track_int', handleInvalid='keep')]
    pipe = Pipeline(stages=indexers).fit(train)
    
    train = pipe.transform(train)
    val = pipe.transform(val)
    test = pipe.transform(test)
    #train.show()

    # Get set of tracks per user
    train_tracks = train.groupBy('user_int').agg(F.collect_set('track_int').alias('tracks'))
    val_tracks = val.groupBy('user_int').agg(F.collect_set('track_int').alias('tracks'))
    test_tracks = test.groupBy('user_int').agg(F.collect_set('track_int').alias('tracks'))
    #train_tracks.show(truncate=False)

    # Get top x tracks -- StringIndexer already sorts by frequency...
    x = 100
    #top_train_tracks = train.groupBy('track_int').agg(F.count('user_int').alias('track_count')).sort(F.desc('track_count')).limit(x)
    #val_train_tracks = val.groupBy('track_int').agg(F.count('user_int').alias('track_count')).sort(F.desc('track_count')).limit(x)

    #top_train_tracks.show()
    top_tracks = np.arange(x).tolist()

    # Getting train MAP...
    map_df = train_tracks.withColumn('recommendations',F.array([F.lit(num) for num in top_tracks]))
    to_int = F.udf(lambda x: list(map(lambda y: int(y),x)))
    joined = map_df.withColumn('tracks',to_int(map_df['tracks']))
    # Add new column that combines rec and tracks
    joined = joined.withColumn('combined',F.struct(joined['recommendations'],joined['tracks']))
    joined.show()
    # Turn into RDD
    combined = joined.select('combined').rdd.map(lambda x:x[0])
    combined = combined.map(lambda line: tuple([ast.literal_eval(x) if isinstance(x, str) else x for x in line]))
    #print(combined.take(5))
    # Get MAP
    metrics = RankingMetrics(combined)
    meanap = metrics.meanAveragePrecision
    print(f'Train MAP: {meanap}')

    # Getting val MAP...
    map_df = val_tracks.withColumn('recommendations',F.array([F.lit(num) for num in top_tracks]))
    to_int = F.udf(lambda x: list(map(lambda y: int(y),x)))
    joined = map_df.withColumn('tracks',to_int(map_df['tracks']))
    # Add new column that combines rec and tracks
    joined = joined.withColumn('combined',F.struct(joined['recommendations'],joined['tracks']))
    joined.show()
    # Turn into RDD
    combined = joined.select('combined').rdd.map(lambda x:x[0])
    combined = combined.map(lambda line: tuple([ast.literal_eval(x) if isinstance(x, str) else x for x in line]))
    #print(combined.take(5))
    # Get MAP
    metrics = RankingMetrics(combined)
    meanap = metrics.meanAveragePrecision
    print(f'Val MAP: {meanap}')
    
    # Getting test MAP...
    map_df = test_tracks.withColumn('recommendations',F.array([F.lit(num) for num in top_tracks]))
    to_int = F.udf(lambda x: list(map(lambda y: int(y),x)))
    joined = map_df.withColumn('tracks',to_int(map_df['tracks']))
    # Add new column that combines rec and tracks
    joined = joined.withColumn('combined',F.struct(joined['recommendations'],joined['tracks']))
    joined.show()
    # Turn into RDD
    combined = joined.select('combined').rdd.map(lambda x:x[0])
    combined = combined.map(lambda line: tuple([ast.literal_eval(x) if isinstance(x, str) else x for x in line]))
    #print(combined.take(5))
    # Get MAP
    metrics = RankingMetrics(combined)
    meanap = metrics.meanAveragePrecision
    print(f'Test MAP: {meanap}')
if __name__ == "__main__":
    # Create the spark session
    spark = SparkSession.builder.appName('baseline').getOrCreate()
    # Get user netID from the command line
    netID = getpass.getuser()
    # Call our main routine
    frac = float(sys.argv[1])
    main(spark, netID, frac)

