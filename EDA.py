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
    # Load in the data files
    sc = spark.sparkContext
    train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet').sample(frac)
    train = train.toDF('user_id','count','track_id', '__index_col_0__').drop('__index_col_0__').repartition(20)
    train.createOrReplaceTempView('train')
    val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet').toDF('user_id','count','track_id','__index_col_0__').drop('__index_col_0__').repartition(20)
    val.createOrReplaceTempView('val')

    # Convert string columns to int columns for use with ALS
    indexers = [StringIndexer(inputCol='user_id', outputCol='user_int', handleInvalid='keep'),
                StringIndexer(inputCol='track_id', outputCol='track_int', handleInvalid='keep')]
    pipe = Pipeline(stages=indexers).fit(train)
    
    # Transform data, drop original columns
    train = pipe.transform(train)
    val = pipe.transform(val)
    train = train.drop('user_id', 'track_id')
    val = val.drop('user_id', 'track_id')  
    
    # Determine track popularity
    train_track_popularity = (train.groupBy('track_int').agg(countDistinct('user_int').alias('num_users')).orderBy('num_users',
                                                                                                             ascending=True))
    val_track_popularity = (val.groupBy('track_int').agg(countDistinct('user_int').alias('num_users')).orderBy('num_users',
                                                                                                             ascending=True))
    print('Number of total tracks in train:', train_track_popularity.count())
    print('Number of total tracks in val:', val_track_popularity.count())
    
    # Determine user activity
    train_user_activity = (train.groupBy('user_int').agg(countDistinct('track_int').alias('num_songs')).orderBy('num_songs',
                                                                                                          ascending=True))
    val_user_activity = (val.groupBy('user_int').agg(countDistinct('track_int').alias('num_songs')).orderBy('num_songs',
                                                                                                          ascending=True))    
    print('Number of total users in train:', train_user_activity.count())    
    print('Number of total users in val:', val_user_activity.count())
    
    # Drop tracks in train that only have one user who listened to the song
    to_drop = train_track_popularity.filter(col('num_users')==1)
    print('Number of songs in train with just one listener:', to_drop.count())
    new_train = train.join(to_drop, 'track_int', how='left_anti')
    
    # Verify that the dropping worked
    new_train_pop = (new_train.groupBy('track_int').agg(countDistinct('user_int').alias('num_users')).orderBy('num_users',
                                                                                                             ascending=True))
    new_train_pop.show(20)
    print('Number of total tracks in new_train_pop:', new_train_pop.count())
    

    
    
if __name__ == "__main__":
    # Create the spark session
    spark = SparkSession.builder.appName('part1').getOrCreate()
    # Get user netID from the command line
    netID = getpass.getuser()
    # Call our main routine
    frac = float(sys.argv[1])
    main(spark, netID, frac)