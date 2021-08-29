from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.functions import col, explode, monotonically_increasing_id, struct, udf, collect_set, count, sum, countDistinct, udf, collect_set
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RankingMetrics
import getpass
import sys
from pyspark.sql.types import LongType
import ast
from itertools import product
import numpy as np
import time


def main(spark, netID, frac):
    sc = spark.sparkContext
    train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet').sample(frac)
    train = train.toDF('user_id','count','track_id', '__index_col_0__').drop('__index_col_0__').repartition(20)
    train.createOrReplaceTempView('train')
    val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet').toDF('user_id','count','track_id','__index_col_0__').drop('__index_col_0__').repartition(20)
    val.createOrReplaceTempView('val')
    test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet').toDF('user_id','count','track_id','__index_col_0__').drop('__index_col_0__').repartition(20)
    test.createOrReplaceTempView('test')

    # Convert string columns to int columns for use with ALS
    indexers = [StringIndexer(inputCol='user_id', outputCol='user_int', handleInvalid='keep'),
                StringIndexer(inputCol='track_id', outputCol='track_int', handleInvalid='keep')]
    pipe = Pipeline(stages=indexers).fit(train)
    
    # Transform data, drop original columns
    train = pipe.transform(train)
    val = pipe.transform(val)
    test = pipe.transform(test)
    train = train.drop('user_id', 'track_id')
    val = val.drop('user_id', 'track_id')
    test = test.drop('user_id', 'track_id')    
    
    
    # Determine track popularity
    train_track_popularity = (train.groupBy('track_int').agg(countDistinct('user_int').alias('num_users')).orderBy('num_users',
                                                                                                             ascending=True))
    val_track_popularity = (val.groupBy('track_int').agg(countDistinct('user_int').alias('num_users')).orderBy('num_users',
                                                                                                             ascending=True)) 
        
    # Drop tracks in train that only have one user who listened to the song
    to_drop = train_track_popularity.filter(col('num_users')==1)
    new_train = train.join(to_drop, 'track_int', how='left_anti')

    # Get set of tracks per user
    train_tracks = new_train.groupBy('user_int').agg(collect_set('track_int').alias('t_tracks'),
                                                 countDistinct('track_int').alias('t_num_tracks'))
    val_tracks = val.groupBy('user_int').agg(collect_set('track_int').alias('v_tracks'), 
                                             countDistinct('track_int').alias('v_num_tracks'))
    

    # Fit tuned model on test set
    als = ALS(rank=200, maxIter=10, regParam=0.1, alpha=50, userCol='user_int', itemCol='track_int',
              ratingCol='count', implicitPrefs=True, coldStartStrategy='drop', numUserBlocks=100, numItemBlocks=100,
              checkpointInterval=-1)
    model = als.fit(new_train)
    preds = model.recommendForUserSubset(test.select('user_int'), numItems=500)
    
    # User int: recommended (ranked) tracks -- we don't need scores for MAP
    tracks_only = udf(lambda x: list(map(lambda y: int(y.track_int), x)))
    preds = preds.withColumn('recommendations', tracks_only(preds['recommendations']))

    # Use left join
    joined = preds.join(test_tracks, 'user_int', 'left')
    to_int = udf(lambda x: list(map(lambda y: int(y),x)))
    joined = joined.withColumn('tracks',to_int(joined['t_tracks']))

    # Add new column that combines recommendations and tracks
    joined = joined.withColumn('combined', struct(joined['recommendations'], joined['tracks']))

    # Turn into RDD
    combined = joined.select('combined').rdd.map(lambda x:x[0])
    combined = combined.map(lambda line: tuple([ast.literal_eval(x) for x in line]))

    # Get MAP
    metrics = RankingMetrics(combined)
    meanap = metrics.meanAveragePrecision
    print(f'MAP: {meanap}')
    

    
    
if __name__ == "__main__":
    # Create the spark session
    spark = SparkSession.builder.appName('part1').getOrCreate()
    # Get user netID from the command line
    netID = getpass.getuser()
    # Call our main routine
    frac = float(sys.argv[1])
    main(spark, netID, frac)