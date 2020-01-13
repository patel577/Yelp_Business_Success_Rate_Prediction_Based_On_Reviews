'''
This script has divided into four parts. 
1. Data Analysis
2. Make profiels of business, customers and reviews profile 
3. Train and Test Machine Learning Model

By: Lovekumar Patel
Aug - Oct 2019
'''

import findspark
from pyspark import SparkContext as sc
import pandas as pd
import os
import numpy as np
import seaborn as sns
import nltk
import json
import pyspark
from nltk.corpus import stopwords
from pyspark.sql import SQLContext
from google.colab import drive
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import MatrixFactorizationModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

data_path = ['../Data']
Business_filepath = os.sep.join(data_path + ['business.json'])
sc = SparkSession.builder.master("local[*]").getOrCreate()
sc = sc(appName="Yelp")
sqlContext = SQLContext(sc)

############################################################################
###################### PART-1: Data Analysis ###############################
############################################################################

#Load Business data
Business_data = sqlContext.read.json(Business_filepath)

### Estimate Average review count and stars by city and category 
#Filter out the attributes that you need
Business = Business_data.select(pyspark.sql.functions.explode(Business_data.categories).alias("category"), 
                               Business_data.state,  Business_data.city, Business_data.stars, Business_data.review_count)
#Register as temp table
Business.registerTempTable("Business_Agg")

#Run the SQL Query
result = sqlContext.sql("SELECT Business_Agg.city, Business_Agg.category,\
AVG(Business_Agg.review_count) As AverageReview,\
AVG(Business_Agg.stars) as AverageStars FROM Business_Agg GROUP BY Business_Agg.city, Business_Agg.category")

#saving the result in a csv file
result.coalesce(1).write.format('com.databricks.spark.csv').option("header", "true").save('Question1')

# Filtering data based on attribute values
#Flatten the category array
CategoryExplo = Business_data.select(pyspark.sql.functions.explode(Business_data.categories).alias("category"),\
                                    Business_data.attributes, Business_data.stars)

#Filter out Mexican and TakeOut
CategoryAtt = CategoryExplo.select(CategoryExplo.attributes.RestaurantsTakeOut.alias("takeout"),\
                                   CategoryExplo.category, CategoryExplo.stars)

CategoryAtt.registerTempTable("CategoryAtt")

#Run the query on the table
MexicanTakeout = "SELECT category, AVG(stars) AS Stars FROM CategoryAtt WHERE category = 'Mexican' \
AND takeout = True GROUP BY category"
RatingMexicanTakeO = sqlContext.sql(MexicanTakeout)

RatingMexicanTakeO.coalesce(1).write.format('com.databricks.spark.csv').option("header", "true").save('Question3')



#### Data Visualization: Geospacial Analysis 
#### Remove the businesses from area where they're highly related and density is very highly

#lattitude, longitude, exploded(category)
LatLong = Business_data.select(pyspark.sql.functions.explode(Business_data.categories).alias("category"), 
                              Business_data.latitude, Business_data.longitude, Business_data.stars,
                              Business_data.review_count)

#Register Temp Table
LatLong.registerTempTable("LatLong")

#Then run a SQL query to filter out the ones only the ones within 15km distance from Toronto Center
BusinessNearTorontofilt = sqlContext.sql("SELECT * FROM  LatLong WHERE \
acos(sin(0.763782941288) * sin(LatLong.latitude * 3.14159 /180) + \
cos(0.763782941288) * cos(LatLong.latitude * 3.14159 /180) * cos((LatLong.longitude * 3.14159 /180) \
- (-1.38598479111))) * 6371 <= 15")

#Then apply the aggregate function on BusinessNearTorontofilt.
BusinessNearTorontofilt.registerTempTable("BusinessNearToronto")
AggQuery = sqlContext.sql("SELECT category,AVG(BusinessNearToronto.stars) AS stars_avg,\
AVG(BusinessNearToronto.review_count) AS Review_count_avg FROM BusinessNearToronto GROUP BY category ORDER BY CATEGORY")

noofBusinesses = AggQuery.count()

AggQuery.coalesce(1).write.format('com.databricks.spark.csv').option("header", "true").save('Question4')





##################################################################################
######################## PROFILE MAKING ##########################################
##################################################################################

## Reviews 
review_file = "/content/yelp_academic_dataset_review.json"
review_raw_rdd = sc.textFile(review_file)
data = review_raw_rdd.map(lambda line : json.loads(line))
review_ids = data.map(lambda line : (line['user_id'],line['business_id'],line['stars'])).cache()
reviews = review_ids.map(lambda x : (x[0],x[2])).groupByKey()

## UserProfiles 
user_file = "/content/yelp_academic_dataset_user.json"
user_review_RDD = sc.textFile(user_file)
data = user_review_RDD.map(lambda line : json.loads(line))
user_ids = data.map(lambda line : (line['user_id'],(line['name'],line['friends']))).cache()

## Business Profiles 
business_file = "/content/yelp_academic_dataset_business.json"
business_review_RDD = sc.textFile(business_file)
business_data = business_review_RDD.map(lambda line : json.loads(line))
business_ids = business_data.map(lambda line :  (line['business_id'],(line['name'], line['address'],line['categories'],line['state'],line['city'], line['latitude'],line['longitude'],line['stars']))).cache()


def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
def get_counts_and_avg(k):
    nratings = len(k[1])
    return k[0], (nratings, float(sum(x for x in k[1])) / nratings)    

k = r.map(get_counts_and_avg).cache()

business_rating_counts_RDD = k.map(lambda x : (x[0],x[1][0])).cache()
avg_usr_ratings = k.map(lambda x : (x[0],x[1][1])).cache()

#####################################################################################
########################## ML Model: linear regeression  ############################
#####################################################################################

rank = 8
seed = 5
iterations = 10
regularization_parameter = 0.1
review_ids_map = review_ids.map(lambda x: (x[0], (x[1], x[2])))            
int_user_id_to_string = user_ids.map(lambda x: x[0]).distinct().zipWithUniqueId().cache()  
int_business_id_to_string = business_ids.map(lambda x: x[0]).distinct().zipWithUniqueId().cache()   
reverse_mapping_user_ids = int_user_id_to_string.map(lambda x: (x[1], x[0]))
reverse_mapping_business_ids = int_business_id_to_string.map(lambda x: (x[1], x[0])) 
review_ids_map = review_ids.map(lambda x: (x[0], (x[1], x[2])))            
user_join_review = review_ids_map.join(int_user_id_to_string).map(lambda x : (x[1][1] , x[1][0]))
ratings = int_business_id_to_string.join(user_join_review.map(lambda x: (x[1][0], (x[0], x[1][1])))).map(lambda x: ( x[1][1][0],x[1][0], x[1][1][1])) 
training_RDD,validation_RDD,test_RDD = ratings.randomSplit([6,2,2],seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x : (x[0],x[1]))
test_for_predict_RDD = test_RDD.map(lambda x : (x[0],x[1]))

model = ALS.train(ratings,rank,seed=5,iterations=10,lambda_=regularization_parameter)
drive.mount("/content/drive")

modelPath = "/content" + "/kq1"
model.save(sc,modelPath)

######################################################################################
######################## Matrix Factorization Model ###################################
######################################################################################

model1 = MatrixFactorizationModel.load(sc , "/content/kq2/kq1")

predictions = model1.predictAll(validation_for_predict_RDD).map(lambda r : ((r[0],r[1]),r[2]))
rates_and_preds = validation_RDD.map(lambda r : (((int(r[0]) , int(r[1])) , float(r[2]) ))).join(predictions)

# Model Evaluation
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
predictions = model1.predictAll(validation_for_predict_RDD)
pred1 = predictions.map(lambda x : (x[0],(x[1],x[2])))
user_ids_to_string_replaced = reverse_mapping_user_ids.join(pred1)   #replacing user ids
replace_both = user_ids_to_string_replaced.keyBy(lambda x: x[1][1][0]).join(reverse_mapping_business_ids).map(lambda x: (x[1][0][1][0],x[1][1],x[1][0][1][1][1]))


user_id1 = replace_both.map(lambda x : (x[0],(x[1],x[2]))).join(user_ids).map(lambda x : (x[1][1][0],x[1][0]))
business_id1 = user_id1.map(lambda x : (x[1][0],(x[0],x[1][1]))).join(business_ids).map(lambda x : (x[1][0][0],x[1][1][0],x[1][0][1]))


#from pyspark.sql.functions import broadcast
#y1 = 'Chris'
#user_name1 = user_ids.filter(lambda user1 : user1[1][0]==y1).map(lambda x : x[0])
#user_id1 =   int_user_id_to_string.filter(lambda user2 : user2[0]==user_name1).map(lambda x : x[1])
#rev1 = int_user_id_to_string.map(lambda x : (x[0],(x[1]))).join(user_id1)
user_id = 80729
user_unrated_business_RDD = review_ids.filter(lambda rating: not rating[0] == user_id ).map(lambda x : (user_id,x[1])).distinct()


user_unrated_business_RDD.join(int_business_id_to_string.map(lambda x :(x[1],x[0]))).take(1)

user_ids_replace = int_user_id_to_string.map(lambda x : (x[0],(x[1]))).join(user_unrated_business_RDD.map(lambda x : (x[1],(x[0])))).map(lambda x : (x[1][0],x[1][1]))
requested_ids =    int_business_id_to_string.keyBy(lambda x: x[0]).rightOuterJoin(user_ids_replace.map(lambda x: (x[1], x[0]))).map(lambda x : (x[1][1], x[1][0][1]))     
user_unrated_business_intids_rdd = requested_ids
predict_ratings = model1.predictAll(user_unrated_business_intids_rdd)
user_ids_business_ids_int=predict_ratings.map(lambda x: (x[0],(x[1],x[2])))     #converting int id to string
user_ids_to_string_replaced = reverse_mapping_user_ids.join(user_ids_business_ids_int);
replace_both = user_ids_to_string_replaced.keyBy(lambda x: x[1][1][0]).join(self.reverse_mapping_business_ids).map(lambda x: (x[1][0][1][0],x[1][1],x[1][0][1][1][1]))       
predict_ratings_string = replace_both
user_names = user_ids.map(lambda x:(x[0],x[1][0]))
business_names = business_ids.map(lambda x:(x[0],(x[1][0],x[1][1])))
predict_ratings_string=predict_ratings_string.map(lambda x: (x[0],(x[1],x[2]))).join(user_names).keyBy(lambda x:x[1][0][0]).join(business_names).map(lambda x:(x[1][0][0],x[0],x[1][0][1][0][1],x[1][0][1][1],x[1][1][0],x[1][1][1])).cache()     
print(predict_ratings_string.take(10))
ratings = predict_ratings_string.filter(lambda r: r[2]>=3).takeOrdered(count,key = lambda x: -x[2])



######################################################################
################# Few Examples For Prediction ########################
######################################################################

predictions = sc.parallelize(model1.recommendProducts(80729,10))     #recommended products for user id 80729
pred1 = predictions.map(lambda x : (x[0],(x[1],x[2])))
user_ids_to_string_replaced = reverse_mapping_user_ids.join(pred1)   #replacing user ids
replace_both = user_ids_to_string_replaced.keyBy(lambda x: x[1][1][0]).join(reverse_mapping_business_ids).map(lambda x: (x[1][0][1][0],x[1][1],x[1][0][1][1][1]))       
user_id2 = replace_both.map(lambda x : (x[0],(x[1],x[2]))).join(user_ids).map(lambda x : (x[1][1][0],x[1][0]))
business_id2 = user_id2.map(lambda x : (x[1][0],(x[0],x[1][1]))).join(business_ids).map(lambda x : (x[1][0][0],x[1][1][0]))
#sc.parallelize(model1.recommendProducts(80729,10)).map(lambda x : (x[0],x[1])).take(10)



predictions = sc.parallelize(model1.recommendProducts(765206,10))     #recommended products for user id 80729
pred1 = predictions.map(lambda x : (x[0],(x[1],x[2])))
user_ids_to_string_replaced = reverse_mapping_user_ids.join(pred1)   #replacing user ids
replace_both = user_ids_to_string_replaced.keyBy(lambda x: x[1][1][0]).join(reverse_mapping_business_ids).map(lambda x: (x[1][0][1][0],x[1][1],x[1][0][1][1][1]))       
user_id2 = replace_both.map(lambda x : (x[0],(x[1],x[2]))).join(user_ids).map(lambda x : (x[1][1][0],x[1][0]))
business_id2 = user_id2.map(lambda x : (x[1][0],(x[0],x[1][1]))).join(business_ids).map(lambda x : (x[1][0][0],x[1][1][0]))
#sc.parallelize(model1.recommendProducts(80729,10)).map(lambda x : (x[0],x[1])).take(10)
