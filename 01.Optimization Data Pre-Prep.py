# Databricks notebook source
from matplotlib import pyplot as plt
import numpy as np
from numpy import sum
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import functions as sf
from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from sklearn.linear_model import LinearRegression
from datetime import datetime
from pyspark.sql.window import Window
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import requests, json, logging, sys
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

# COMMAND ----------

#SPN connection interface
def getToken(sf_spn_client_id, sf_spn_client_secret):
    host = "https://login.microsoftonline.com/"
    url = "12a3af23-a769-4654-847f-958f3d479f4a/oauth2/v2.0/token"
    headers = { 'Content-Type': 'application/x-www-form-urlencoded' }
    payload = 'client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope=https://aad20912-2ec3-409d-8995-91d64894d38a/.default'.format(client_id=sf_spn_client_id,client_secret=sf_spn_client_secret)
    
    auth_token_request = requests.request("POST", host + url, headers=headers, data=payload)
    auth_response = json.loads(auth_token_request.text)
    return auth_response['access_token']
  
# Client Id of Service Principle 
spn_clientId = 'f85542e4-f419-49ec-9c29-ad875e32dde1'
# spn_clientId = '73cf19ce-ab03-41cb-a7af-f6bccf41f88f'
# Client Secret of Service Principle 
# spn_clientSecret = 'Pb78Q~NyxLU6TFe.ZCiD8rJeTICTCCg1AZgocbj2' 
spn_clientSecret = 'oHv8Q~BZ3ul.jOYDifiTIdEuZLv9cf1EdPUfyajt' 

auth_token = getToken(spn_clientId, spn_clientSecret)
#Access Snowflake in databricks environment to read data tables
options = {
            "sfUrl": "https://nestleprd.west-europe.privatelink.snowflakecomputing.com",
            "sfUser": "390877d5-c339-40a3-99ad-8d4ffe306c14",
            "sfDatabase": "JP",
            "sfAuthenticator" : "oauth",
            "sfToken" : auth_token,            
            "sfSchema": "PRS",
            "sfWarehouse": "JP_MIDJP_VW1"
          }

# COMMAND ----------

# %sql 
# CREATE DATABASE ui_output_v32

# COMMAND ----------

decomp_database_name = 'tpo_output_v32'
print(decomp_database_name)
opt_database_name = 'ui_output_v32'
print(opt_database_name)

# decomp_database_name = 'tpo_output_v19'
# print(decomp_database_name)
# opt_database_name = 'ui_output_v14'
# print(opt_database_name)

# decomp_database_name = 'tpo_output_v18'
# print(decomp_database_name)
# opt_database_name = 'ui_output_v13'
# print(opt_database_name)

# decomp_database_name = 'tpo_output_v15'
# print(decomp_database_name)
# opt_database_name = 'ui_output_v10'
# print(opt_database_name)

# decomp_database_name = 'tpo_output_v12'
# print(decomp_database_name)
# opt_database_name = 'ui_output_v7'
# print(opt_database_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing the Optimization Input

# COMMAND ----------

w = Window.partitionBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
decomp_data = (spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME not in ('Base_BasePrice','Volume','PREDICTED_SALES')")
               .distinct())
# print(decomp_data.columns)

# COMMAND ----------

decomp_data = (
    decomp_data.drop("DATE", "HALF_YEAR", "MC", "EC", "IND")
    .distinct()
    .withColumn("TOTAL_SPENDS", sf.sum("SPENDS").over(w))
    .withColumn("TOTAL_VOLUME", sf.sum("VOLUME").over(w))
    .withColumn("TOTAL_GPS", sf.sum("GPS").over(w))
    .withColumn("TOTAL_NETSALES", sf.sum("NETSALES").over(w))
    .withColumn(
        "SPENDS_CONTRI",
        sf.when(
            sf.col("TOTAL_SPENDS") != 0,
            sf.round(sf.col("SPENDS") / sf.col("TOTAL_SPENDS"), 3),
        ).otherwise(sf.lit(0)),
    )
    #    .withColumn('SPENDS_CONTRI',sf.when(((sf.col('TOTAL_SPENDS')!=0) & (sf.col('MODELINGNAME').isin('PROMOTIONAD', 'LIQUIDATION', 'PRICEOFF'))),sf.round(sf.col('SPENDS')/sf.col('TOTAL_SPENDS'),3)).otherwise(sf.lit(0)))
    .withColumn(
        "VOL_CONTRI",
        sf.when(
            sf.col("TOTAL_VOLUME") != 0,
            sf.round(sf.col("VOLUME") / sf.col("TOTAL_VOLUME"), 3),
        ).otherwise(sf.lit(0)),
    )
    .withColumn(
        "GPS_CONTRI",
        sf.when(
            sf.col("TOTAL_GPS") != 0, sf.round(sf.col("GPS") / sf.col("TOTAL_GPS"), 3)
        ).otherwise(sf.lit(0)),
    )
    .withColumn(
        "NS_CONTRI",
        sf.when(
            sf.col("TOTAL_NETSALES") != 0,
            sf.round(sf.col("NETSALES") / sf.col("TOTAL_NETSALES"), 3),
        ).otherwise(sf.lit(0)),
    )
    .withColumn("PRICE", sf.col("GPS")/sf.col("VOLUME"))  # added
    .orderBy(
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "SEGMENT",
        "SHISHA",
        "CLUSTER",
        "CHANNEL",
        "PLANNINGCUSTOMER",
        "YEAR",
        "MONTH",
    )
    .drop(
        "VOLUME",
        "TOTAL_VOLUME",
        "SPENDS",
        "TOTAL_SPENDS",
        "GPS",
        "TOTAL_GPS",
        "NETSALES",
        "TOTAL_NETSALES",
    )
    .distinct()
)
decomp_data.display()

# COMMAND ----------

# decomp_data_1 = (decomp_data
#             #    .drop('DATE', 'HALF_YEAR', 'MC', 'EC', 'IND')
#                .distinct()
#                .withColumn('TOTAL_SPENDS', sf.sum('SPENDS').over(w))
#                .withColumn('TOTAL_VOLUME', sf.sum('VOLUME').over(w))
#                .withColumn('TOTAL_GPS', sf.sum('GPS').over(w))
#                .withColumn('TOTAL_NETSALES', sf.sum('NETSALES').over(w))
#                .withColumn('SPENDS_CONTRI',sf.when(sf.col('TOTAL_SPENDS')!=0,sf.round(sf.col('SPENDS')/sf.col('TOTAL_SPENDS'),3)).otherwise(sf.lit(0)))
#                .withColumn('VOL_CONTRI',sf.when(sf.col('TOTAL_VOLUME')!=0, sf.round(sf.col('VOLUME')/sf.col('TOTAL_VOLUME'),3)).otherwise(sf.lit(0)))
#                .withColumn('GPS_CONTRI',sf.when(sf.col('TOTAL_GPS')!=0, sf.round(sf.col('GPS')/sf.col('TOTAL_GPS'),3)).otherwise(sf.lit(0)))
#                .withColumn('NS_CONTRI',sf.when(sf.col('TOTAL_NETSALES')!=0, sf.round(sf.col('NETSALES')/sf.col('TOTAL_NETSALES'),3)).otherwise(sf.lit(0)))
#                .orderBy('BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER','YEAR', 'MONTH')
#             #    .drop('VOLUME','TOTAL_VOLUME','SPENDS','TOTAL_SPENDS','GPS','TOTAL_GPS','NETSALES','TOTAL_NETSALES')
#                .distinct()
#                )
# decomp_data.display()

# COMMAND ----------

# decomp_data.filter(sf.col("DATE") >= "2024-01-01 00:00:00").filter(sf.col("DATE") <= "2024-06-30 23:59:59").groupBy("SEGMENT").agg(sf.sum('NETSALES')).display()

# decomp_data.filter(sf.col("YEAR") == 2024).filter(sf.col("MONTH")==6).filter(sf.col("SEGMENT").isin("JP/CNF/In To Home/KKLE HR/M-bag")).display()
decomp_data.select('MODELINGNAME').distinct().collect()
# decomp_data.filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").groupBy("YEAR","MONTH").agg(sf.sum("NETSALES")).display()

# COMMAND ----------

actual_data = (spark.sql("select * from {0}.mbt_wo_competition_price_corrected".format(decomp_database_name))
               .distinct()
               )
# print(actual_data.select(sf.sum('SALESAMT')).collect()[0][0])
# print(actual_data.select(sf.sum('SALESQTYCASE')).collect()[0][0])
actual_data = (actual_data
               .withColumn('SPENDS',sf.col('PRICEOFF')+sf.col('PROMOTIONAD')+sf.col('LIQUIDATION'))
               .groupBy('YEAR','MONTH','SEGMENT','SHISHANM','PLANNINGCUSTOMER')
               .agg(sf.sum('SPENDS').alias('TOTAL_SPENDS'),
                    sf.sum('SALESAMT').alias('TOTAL_SALESAMT'),
                    sf.sum('SALESQTYCASE').alias('TOTAL_SALESQTYCASE'))
               .withColumn('YEAR',sf.col('YEAR').cast('integer'))
               .withColumn('MONTH',sf.col('MONTH').cast('integer'))
               .withColumnRenamed('SHISHANM','SHISHA')
               .distinct()
               )
print(actual_data.select(sf.sum('TOTAL_SPENDS')).collect()[0][0])
print(actual_data.select(sf.sum('TOTAL_SALESAMT')).collect()[0][0])
print(actual_data.select(sf.sum('TOTAL_SALESQTYCASE')).collect()[0][0])
print(actual_data.count())
actual_data.display()

# COMMAND ----------

new_data = (decomp_data
            .join(actual_data,['YEAR', 'MONTH', 'SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'],'left')
            .distinct()
            .withColumn('VOLUME',sf.col('VOL_CONTRI')*sf.col('TOTAL_SALESQTYCASE'))
            .withColumn('SPENDS',sf.col('SPENDS_CONTRI')*sf.col('TOTAL_SPENDS'))
            # .withColumn('GPS',sf.col('GPS_CONTRI')*sf.col('TOTAL_SALESAMT'))
            .withColumn('GPS',sf.col('VOLUME')*sf.col('PRICE'))
            .fillna(0, subset=["SPENDS","VOLUME","GPS"])
            .withColumn('EC',sf.col('EC_PER_QTYCASE')*sf.col('VOLUME'))
            # .withColumn('TOTAL_GPS', sf.sum('GPS').over(w))
            .withColumn('TOTAL_NETSALES', sf.col('TOTAL_SALESAMT')-sf.col('TOTAL_SPENDS'))
            # .withColumn('NETSALES', sf.col('NS_CONTRI')*sf.col('TOTAL_NETSALES'))
            .withColumn('NETSALES', sf.col('GPS')-sf.col('SPENDS'))
            .withColumn('MC', sf.col('NETSALES')-sf.col('EC'))
            # .drop('SPENDS_CONTRI','VOL_CONTRI','TOTAL_NETSALES','TOTAL_SPENDS','TOTAL_SALESQTYCASE','NS_CONTRI', 'TOTAL_SALESAMT','GPS_CONTRI')
            # .select('YEAR', 'MONTH','BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER','MODELINGNAME','SPENDS','NETSALES','GPS','VOLUME','EC', 'MC','PRICE','EC_PER_QTYCASE')
            .orderBy('BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER','YEAR', 'MONTH')
            )
col_list = ['SPENDS','NETSALES','GPS','VOLUME','EC', 'MC','PRICE','EC_PER_QTYCASE']
for col in col_list:
    new_data = new_data.withColumn(col,sf.round(sf.col(col),3))

print(new_data.select(sf.sum('SPENDS')).collect()[0][0])
print(new_data.select(sf.sum('GPS')).collect()[0][0])
print(new_data.select(sf.sum('NETSALES')).collect()[0][0])
print(new_data.select(sf.sum('VOLUME')).collect()[0][0])
# new_data.display()

# COMMAND ----------

new_data.filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").display()

# COMMAND ----------

# new_data = (decomp_data
#             .join(actual_data,['YEAR', 'MONTH', 'SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'],'left')
#             .distinct()
#             .withColumn('VOLUME',sf.col('VOL_CONTRI')*sf.col('TOTAL_SALESQTYCASE'))
#             .withColumn('SPENDS',sf.col('SPENDS_CONTRI')*sf.col('TOTAL_SPENDS'))
#             .withColumn('GPS',sf.col('GPS_CONTRI')*sf.col('TOTAL_SALESAMT'))
#             .withColumn('EC',sf.col('EC_PER_QTYCASE')*sf.col('VOLUME'))
#             # .withColumn('TOTAL_GPS', sf.sum('GPS').over(w))
#             .withColumn('TOTAL_NETSALES', sf.col('TOTAL_SALESAMT')-sf.col('TOTAL_SPENDS'))
#             .withColumn('NETSALES', sf.col('NS_CONTRI')*sf.col('TOTAL_NETSALES'))
#             .withColumn('MC', sf.col('NETSALES')-sf.col('EC'))
#             .drop('SPENDS_CONTRI','VOL_CONTRI','TOTAL_NETSALES','TOTAL_SPENDS','TOTAL_SALESQTYCASE','NS_CONTRI', 'TOTAL_SALESAMT','GPS_CONTRI')
#             .select('YEAR', 'MONTH','BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER','MODELINGNAME','SPENDS','NETSALES','GPS','VOLUME','EC', 'MC','PRICE','EC_PER_QTYCASE')
#             .orderBy('BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER','YEAR', 'MONTH')
#             )
# col_list = ['SPENDS','NETSALES','GPS','VOLUME','EC', 'MC','PRICE','EC_PER_QTYCASE']
# for col in col_list:
#     new_data = new_data.withColumn(col,sf.round(sf.col(col),3))

# print(new_data.select(sf.sum('SPENDS')).collect()[0][0])
# print(new_data.select(sf.sum('GPS')).collect()[0][0])
# print(new_data.select(sf.sum('NETSALES')).collect()[0][0])
# print(new_data.select(sf.sum('VOLUME')).collect()[0][0])
# new_data.display()

# COMMAND ----------

# new_data.filter(sf.col("YEAR") == 2023).groupBy("SEGMENT").agg(sf.sum("NETSALES").alias('NETSALES')).display()

# COMMAND ----------

new_data = (new_data
            .withColumn('MONTH',sf.col('MONTH').cast('string'))
             .withColumn('YEAR',sf.col('YEAR').cast('string'))
             .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
             )
# new_data.display()

# COMMAND ----------

new_data.filter("PLANNINGCUSTOMER == 'L5 JP NS Lawson Shinagawa' and SEGMENT == 'JP/CNF/In To Home/KKCORE Standard/M-Bag' and MONTH == 7").display()

# COMMAND ----------

table_name = opt_database_name+".optimization_input"
print(table_name)
new_data.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price Change Inclusion

# COMMAND ----------

w = Window.partitionBy('SEGMENT')
waterfall_df = (spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
                .filter("MODELINGNAME not in ('Base_BasePrice','Volume','PREDICTED_SALES')")
                .groupBy('SEGMENT','MODELINGNAME')
                .agg(sf.sum('VOLUME').alias('OWN_PRICE_CONTRI'))
                .orderBy('SEGMENT')
                .withColumn('TOTAL',sf.sum('OWN_PRICE_CONTRI').over(w))
                .withColumn('OWN_PRICE_CONTRI',(sf.col('OWN_PRICE_CONTRI')/sf.col('TOTAL'))*sf.lit(100))
                .filter("MODELINGNAME = 'Own Price'")
                .drop('TOTAL','MODELINGNAME')
                )
waterfall_df.display()

# COMMAND ----------

pc_df = (spark.read.format("snowflake").options(**options).option("dbtable","VW_PRICE_CHANGE").load()
         .withColumn('PRICE_CHANGE',sf.col('PRICE_CHANGE')*sf.lit(100))
         .join(waterfall_df,['SEGMENT'],'inner')
         .withColumn('OWN_PRICE_NEW',(sf.col('PRICE_CHANGE')*sf.col('OWN_PRICE_CONTRI'))/sf.lit(100))
         )
pc_df.display()

# COMMAND ----------

price_updt = (spark.sql("select * from {0}.optimization_input".format(opt_database_name))
              .join((pc_df
                   .select('SEGMENT','PRICE_CHANGE',sf.col('OWN_PRICE_NEW').alias('MULTIPLIER'))
                   .withColumn('PRICE_CHANGE',sf.col('PRICE_CHANGE')/sf.lit(100))
                   ),['SEGMENT'],'left')
              .distinct()
              .fillna(0,subset=['PRICE_CHANGE','MULTIPLIER'])
              .withColumn('PRICE',sf.col('PRICE')*(sf.lit(1)+sf.col('PRICE_CHANGE')))
              )
col_list = ['GPS','VOLUME','NETSALES']
for col in col_list:
    price_updt = (price_updt
                  .withColumn(col,sf.when(sf.col('MODELINGNAME')=='PRICEOFF',sf.col(col)*(sf.lit(1)+sf.col('MULTIPLIER'))).otherwise(sf.col(col)))
                  .withColumn(col,sf.when(sf.col('MODELINGNAME')=='Own Price',sf.col(col)*(sf.lit(1)-sf.col('MULTIPLIER'))).otherwise(sf.col(col)))
                  )
price_updt = price_updt.drop('PRICE_CHANGE','MULTIPLIER').distinct()
price_updt.display()

# COMMAND ----------

table_name = opt_database_name+".optimization_input_pc"
print(table_name)
price_updt.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### HO Branch Reallocation

# COMMAND ----------

mbt_new1 = spark.sql("select * from {0}.mbt_updated_all_1".format(decomp_database_name))
mbt_new2 = spark.sql("select * from {0}.mbt_updated_all_2".format(decomp_database_name))
mbt_new3 = spark.sql("select * from {0}.mbt_updated_all_3".format(decomp_database_name))
mbt_new4 = spark.sql("select * from {0}.mbt_updated_all_4".format(decomp_database_name))

mbt_new = (mbt_new1
           .select('YEARMONTH', 'YEAR', 'MONTH', 'WEEK_START', 'SEGMENTCD', 'SEGMENT', 'SHISHACD', 'SHISHANM', 'OLD_PLANNINGCUSTOMERCD', 'NEW_PLANNINGCUSTOMERCD', 'NEW_PLANNINGCUSTOMER', 'PRICEOFF', 'LIQUIDATION', 'PROMOTIONAD', 'ADD_PRICEOFF2', 'ADD_LIQUIDATION2', 'ADD_PROMOTIONAD2', 'EC', 'SALESAMT', 'SALESWTGRAMS', 'SALESQTYCASE', 'NEW_PRICEOFF', 'NEW_LIQUIDATION', 'NEW_PROMOTIONAD', 'INI_CONTRI_PO', 'ADD_CONTRI_PO', 'INI_CONTRI_LIQ', 'ADD_CONTRI_LIQ', 'INI_CONTRI_PA', 'ADD_CONTRI_PA', 'PERC_CONTRI', 'IND')
           .union(mbt_new2)
           .union(mbt_new3)
           .union(mbt_new4)
           .join((spark.read.format("snowflake").options(**options).option("dbtable","VW_CLEAR_CUSTOMER_MASTER").load()
                  .select(sf.col('PLANNINGCUSTOMERCD').alias('OLD_PLANNINGCUSTOMERCD'),sf.col('PLANNINGCUSTOMER').alias('OLD_PLANNINGCUSTOMER'))
                  .distinct()
                  ),'OLD_PLANNINGCUSTOMERCD','left')
           .filter("SALESAMT != 0 or (NEW_PRICEOFF == 0 and NEW_LIQUIDATION == 0 and NEW_PROMOTIONAD == 0)")
           .withColumn('INI_PRICEOFF', sf.col('INI_CONTRI_PO')*sf.col('NEW_PRICEOFF'))
           .withColumn('ADD_PRICEOFF', sf.col('ADD_CONTRI_PO')*sf.col('NEW_PRICEOFF'))
           .withColumn('INI_LIQUIDATION', sf.col('INI_CONTRI_LIQ')*sf.col('NEW_LIQUIDATION'))
           .withColumn('ADD_LIQUIDATION', sf.col('ADD_CONTRI_LIQ')*sf.col('NEW_LIQUIDATION'))
           .withColumn('INI_PROMOTIONAD', sf.col('INI_CONTRI_PA')*sf.col('NEW_PROMOTIONAD'))
           .withColumn('ADD_PROMOTIONAD', sf.col('ADD_CONTRI_PA')*sf.col('NEW_PROMOTIONAD'))
           .select('YEARMONTH', 'YEAR', 'MONTH', 'WEEK_START', 'SEGMENT', 'SHISHANM', 'NEW_PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER', 'INI_PRICEOFF', 'ADD_PRICEOFF', 'INI_LIQUIDATION', 'ADD_LIQUIDATION', 'INI_PROMOTIONAD', 'ADD_PROMOTIONAD')
           .distinct()
           )
# print(mbt_new.count())
mbt_new.display()

# COMMAND ----------

(mbt_new
 .withColumn('CATEGORY',sf.when(sf.col('SEGMENT').contains('JP/C&B/RSC'),sf.lit('JP/C&B/RSC'))
                       .otherwise(sf.when(sf.col('SEGMENT').contains('JP/C&B/RTD'),sf.lit('JP/C&B/RTD'))
                                  .otherwise(sf.when(sf.col('SEGMENT').contains('JP/C&B/SS'),sf.lit('JP/C&B/Single Serve'))
                                             .otherwise(sf.lit('JP/CNF/In To Home')))))
 .groupBy('CATEGORY')
 .agg(sf.sum('INI_PRICEOFF').alias('INI_PRICEOFF'),
      sf.sum('ADD_PRICEOFF').alias('ADD_PRICEOFF'),
      sf.sum('INI_LIQUIDATION').alias('INI_LIQUIDATION'),
      sf.sum('ADD_LIQUIDATION').alias('ADD_LIQUIDATION'),
      sf.sum('INI_PROMOTIONAD').alias('INI_PROMOTIONAD'),
      sf.sum('ADD_PROMOTIONAD').alias('ADD_PROMOTIONAD'))
 .withColumn('NEW_PRICEOFF',sf.col('INI_PRICEOFF')+sf.col('ADD_PRICEOFF'))
 .withColumn('NEW_LIQUIDATION',sf.col('INI_LIQUIDATION')+sf.col('ADD_LIQUIDATION'))
 .withColumn('NEW_PROMOTIONAD',sf.col('INI_PROMOTIONAD')+sf.col('ADD_PROMOTIONAD'))
 ).display()

# COMMAND ----------

columns = mbt_new.columns
columns.remove('WEEK_START')
mbt_new.select([sf.count(sf.when(sf.isnan(c) | sf.col(c).isNull(), c)).alias(c) for c in columns]).display()

# COMMAND ----------

table_name = opt_database_name + '.ho_branch_info'
print(table_name)
mbt_new.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Updating Historical Data

# COMMAND ----------

table_to_update = 'optimization_input_pc'
ho_branch_table = 'ho_branch_info'

# COMMAND ----------

print(spark.sql("select * from {0}.{1}".format(opt_database_name,ho_branch_table)).columns)
print(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update)).columns)

# COMMAND ----------

no_update = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
             .filter("MODELINGNAME not in ('PRICEOFF','LIQUIDATION','PROMOTIONAD')")
             .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
             .distinct()
             )

# COMMAND ----------

df = (spark.sql("select * from {0}.{1}".format(opt_database_name,ho_branch_table))
      .withColumnRenamed('NEW_PLANNINGCUSTOMER','PLANNINGCUSTOMER')
      .withColumnRenamed('SHISHANM','SHISHA')
      .groupBy('YEAR', 'MONTH', 'SEGMENT','SHISHA','PLANNINGCUSTOMER','OLD_PLANNINGCUSTOMER')
      .agg(sf.sum('INI_PRICEOFF').alias('INI_PRICEOFF'),
           sf.sum('ADD_PRICEOFF').alias('ADD_PRICEOFF'),
           sf.sum('INI_LIQUIDATION').alias('INI_LIQUIDATION'),
           sf.sum('ADD_LIQUIDATION').alias('ADD_LIQUIDATION'),
           sf.sum('INI_PROMOTIONAD').alias('INI_PROMOTIONAD'),
           sf.sum('ADD_PROMOTIONAD').alias('ADD_PROMOTIONAD'))
      .withColumn('TOTAL_PO',sf.col('INI_PRICEOFF')+sf.col('ADD_PRICEOFF'))
      .withColumn('TOTAL_LIQ',sf.col('INI_LIQUIDATION')+sf.col('ADD_LIQUIDATION'))
      .withColumn('TOTAL_PA',sf.col('INI_PROMOTIONAD')+sf.col('ADD_PROMOTIONAD'))
      .withColumn('INI_CONTRI_PO',sf.when(sf.col('TOTAL_PO')!=0,sf.round(sf.col('INI_PRICEOFF')/sf.col('TOTAL_PO'),3)).otherwise(sf.lit(0)))
      .withColumn('ADD_CONTRI_PO',sf.when(sf.col('TOTAL_PO')!=0,sf.round(sf.col('ADD_PRICEOFF')/sf.col('TOTAL_PO'),3)).otherwise(sf.lit(0)))
      .withColumn('INI_CONTRI_LIQ',sf.when(sf.col('TOTAL_LIQ')!=0,sf.round(sf.col('INI_LIQUIDATION')/sf.col('TOTAL_LIQ'),3)).otherwise(sf.lit(0)))
      .withColumn('ADD_CONTRI_LIQ',sf.when(sf.col('TOTAL_LIQ')!=0,sf.round(sf.col('ADD_LIQUIDATION')/sf.col('TOTAL_LIQ'),3)).otherwise(sf.lit(0)))
      .withColumn('INI_CONTRI_PA',sf.when(sf.col('TOTAL_PA')!=0,sf.round(sf.col('INI_PROMOTIONAD')/sf.col('TOTAL_PA'),3)).otherwise(sf.lit(0)))
      .withColumn('ADD_CONTRI_PA',sf.when(sf.col('TOTAL_PA')!=0,sf.round(sf.col('ADD_PROMOTIONAD')/sf.col('TOTAL_PA'),3)).otherwise(sf.lit(0)))
      .drop('INI_PRICEOFF','ADD_PRICEOFF','TOTAL_PO','INI_LIQUIDATION','ADD_LIQUIDATION','TOTAL_LIQ','INI_PROMOTIONAD','ADD_PROMOTIONAD','TOTAL_PA')
      .join((spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
             .filter("(MODELINGNAME in ('PRICEOFF','LIQUIDATION','PROMOTIONAD')) and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
             .distinct()
             ),['YEAR','MONTH','SHISHA','PLANNINGCUSTOMER','SEGMENT'],'right')
      .distinct()
      ).toPandas()
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Priceoff

# COMMAND ----------

priceoff =  (spark.createDataFrame(pd.melt(df[df['MODELINGNAME']=='PRICEOFF'].reset_index(drop=True),
                                    id_vars=['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT',
                                             'OLD_PLANNINGCUSTOMER', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'CLUSTER', 'CHANNEL', 'MODELINGNAME', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME', 'EC', 'MC', 'PRICE', 'EC_PER_QTYCASE'], 
                                    value_vars = ['INI_CONTRI_PO', 'ADD_CONTRI_PO'], 
                                    var_name='CONTRI_TYPE',
                                    value_name='CONTRI_VAL')
                                   )
            #  .filter(sf.col('CONTRI_VAL')!=0)
             .distinct()
             )
# priceoff.display()
print(priceoff.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
priceoff_g1 = (priceoff
               .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .filter("CONTRI_VAL = 0")
               .drop('CONTRI_VAL')
               .join((spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
                      .filter("(MODELINGNAME = 'PRICEOFF') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
                      .distinct()
                      ),['YEAR','MONTH','SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
               )
print(priceoff_g1.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
priceoff_g2 = (priceoff
               .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .filter("CONTRI_VAL != 0")
               .drop('CONTRI_VAL')
               .join(priceoff,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'left')
               .filter(sf.col('CONTRI_VAL')!=0)
               )
print(priceoff_g2.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())

# COMMAND ----------

priceoff_g2.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.collect_set('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()

# COMMAND ----------

print(priceoff_g2.count())
print((priceoff_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter(sf.col('CONTRI_VAL')<2)
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
             ).count())
print((priceoff_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO"))))
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
          #    .filter((sf.col('CONTRI_VAL')!=1))
             .distinct()
             ).count())
print((priceoff_g2
               .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                    sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                    sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
               .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO")))))
               .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .distinct()
               .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
               .distinct()
               ).count())

# COMMAND ----------

po_group_one = (priceoff_g2
                .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter(sf.col('CONTRI_VAL')<2)
                .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .distinct()
                .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                )
# print(po_group_one.count())
po_group_two = (priceoff_g2
                .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO"))))
                .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .distinct()
                .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                .filter((sf.col('CONTRI_VAL')!=1))
                .distinct()
                )
# po_group_two.filter(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER')).display()
# po_group_two.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()
# po_print(group_two.count())
w = Window.partitionBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
po_group_three = (priceoff_g2
                  .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                       sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                       sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                  .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                  .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO")))))
                  .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .distinct()
                  .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                  .withColumn('COUNT',sf.count('YEAR').over(w))
                  .withColumn('CONTRI_VAL',sf.col('CONTRI_VAL')/sf.col('COUNT'))
                  .drop('COUNT')
                  .distinct()
                  )
po_updt = po_group_one.unionByName(po_group_two).distinct().unionByName(po_group_three).distinct()
columns = ['SPENDS', 'NETSALES', 'GPS', 'VOLUME']
for col in columns:
    po_updt = po_updt.withColumn(col,sf.round(sf.col(col)*sf.col('CONTRI_VAL'),3))
po_updt = (po_updt
           .withColumn('PLANNINGCUSTOMER_final',
                       sf.when(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER'),sf.col('PLANNINGCUSTOMER'))
                       .otherwise(sf.when((sf.col('PLANNINGCUSTOMER')!=sf.col('OLD_PLANNINGCUSTOMER'))&(sf.col('CONTRI_TYPE')=='INI_CONTRI_PO'),sf.col('PLANNINGCUSTOMER'))
                                  .otherwise(sf.col('OLD_PLANNINGCUSTOMER'))))
           .drop('PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER')
           .withColumnRenamed('PLANNINGCUSTOMER_final','PLANNINGCUSTOMER')
           .groupBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
           .agg(sf.mean('PRICE').alias('PRICE'),
                sf.sum('SPENDS').alias('SPENDS'),
                sf.sum('NETSALES').alias('NETSALES'),
                sf.sum('GPS').alias('GPS'),
                sf.sum('VOLUME').alias('VOLUME'))
           )
po_updt.display()

# COMMAND ----------

po_updt = po_updt.union(priceoff_g1.select(*po_updt.columns))
print(po_updt.count())

# COMMAND ----------

po_updt.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
 .filter("(MODELINGNAME = 'PRICEOFF') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
 .distinct()
 ).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Liquidation

# COMMAND ----------

liquidation =  (spark.createDataFrame(pd.melt(df[df['MODELINGNAME']=='LIQUIDATION'].reset_index(drop=True),
                                              id_vars=['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT',
                                                       'OLD_PLANNINGCUSTOMER', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'CLUSTER', 'CHANNEL', 'MODELINGNAME', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME', 'EC', 'MC', 'PRICE', 'EC_PER_QTYCASE'], 
                                              value_vars = ['INI_CONTRI_LIQ', 'ADD_CONTRI_LIQ'], 
                                              var_name='CONTRI_TYPE',
                                              value_name='CONTRI_VAL')
                                      )
                # .filter(sf.col('CONTRI_VAL')!=0)
                .distinct()
                )
# liquidation.display()
print(liquidation.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
liquidation_g1 = (liquidation
                  .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                  .filter("CONTRI_VAL = 0")
                  .drop('CONTRI_VAL')
                  .join((spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
                         .filter("(MODELINGNAME = 'LIQUIDATION') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
                         .distinct()
                         ),['YEAR','MONTH','SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
               )
print(liquidation_g1.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
liquidation_g2 = (liquidation
                  .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                  .filter("CONTRI_VAL != 0")
                  .drop('CONTRI_VAL')
                  .join(liquidation,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'left')
                  .filter(sf.col('CONTRI_VAL')!=0)
                  )
print(liquidation_g2.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())

# COMMAND ----------

liquidation_g2.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.collect_set('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()

# COMMAND ----------

print(liquidation_g2.count())
print((liquidation_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter(sf.col('CONTRI_VAL')<2)
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(liquidation_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
             ).count())
print((liquidation_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_LIQ"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_LIQ"))))
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(liquidation_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
          #    .filter((sf.col('CONTRI_VAL')!=1))
             .distinct()
             ).count())
print((liquidation_g2
               .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                    sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                    sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
               .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_LIQ"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_LIQ")))))
               .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .distinct()
               .join(liquidation_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
               .distinct()
               ).count())

# COMMAND ----------

liq_group_one = (liquidation_g2
                 .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                 .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                      sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                      sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                 .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                 .filter(sf.col('CONTRI_VAL')<2)
                 .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                 .distinct()
                 .join(liquidation_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                 )
# print(liq_group_one.count())
liq_group_two = (liquidation_g2
                 .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                 .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                      sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                      sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                 .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                 .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_LIQ"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_LIQ"))))
                 .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                 .distinct()
                 .join(priceoff_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                 .filter((sf.col('CONTRI_VAL')!=1))
                 .distinct()
                 )
# liq_group_two.filter(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER')).display()
# liq_group_two.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()
# liq_print(group_two.count())
w = Window.partitionBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
liq_group_three = (liquidation_g2
                   .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                   .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                        sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                        sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                   .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                   .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_LIQ"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_LIQ")))))
                   .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                   .distinct()
                   .join(liquidation_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                  .withColumn('COUNT',sf.count('YEAR').over(w))
                  .withColumn('CONTRI_VAL',sf.col('CONTRI_VAL')/sf.col('COUNT'))
                  .drop('COUNT')
                  .distinct()
                  )
liq_updt = liq_group_one.unionByName(liq_group_two).distinct().unionByName(liq_group_three).distinct()
columns = ['SPENDS', 'NETSALES', 'GPS', 'VOLUME']
for col in columns:
    liq_updt = liq_updt.withColumn(col,sf.round(sf.col(col)*sf.col('CONTRI_VAL'),3))
liq_updt = (liq_updt
            .withColumn('PLANNINGCUSTOMER_final',
                        sf.when(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER'),sf.col('PLANNINGCUSTOMER'))
                        .otherwise(sf.when((sf.col('PLANNINGCUSTOMER')!=sf.col('OLD_PLANNINGCUSTOMER'))&(sf.col('CONTRI_TYPE')=='INI_CONTRI_PO'),sf.col('PLANNINGCUSTOMER'))
                                   .otherwise(sf.col('OLD_PLANNINGCUSTOMER'))))
            .drop('PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER')
            .withColumnRenamed('PLANNINGCUSTOMER_final','PLANNINGCUSTOMER')
            .groupBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
            .agg(sf.mean('PRICE').alias('PRICE'),
                 sf.sum('SPENDS').alias('SPENDS'),
                 sf.sum('NETSALES').alias('NETSALES'),
                 sf.sum('GPS').alias('GPS'),
                 sf.sum('VOLUME').alias('VOLUME'))
            )
liq_updt.display()

# COMMAND ----------

liq_updt = liq_updt.union(liquidation_g1.select(*liq_updt.columns))
print(liq_updt.count())

# COMMAND ----------

liq_updt.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
 .filter("(MODELINGNAME = 'LIQUIDATION') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
 .distinct()
 ).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promotion Ad

# COMMAND ----------

promoad =  (spark.createDataFrame(pd.melt(df[df['MODELINGNAME']=='PROMOTIONAD'].reset_index(drop=True),
                                          id_vars=['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT',
                                                   'OLD_PLANNINGCUSTOMER', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'CLUSTER', 'CHANNEL', 'MODELINGNAME', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME', 'EC', 'MC', 'PRICE', 'EC_PER_QTYCASE'], 
                                          value_vars = ['INI_CONTRI_PA', 'ADD_CONTRI_PA'], 
                                          var_name='CONTRI_TYPE',
                                          value_name='CONTRI_VAL')
                                  )
            # .filter(sf.col('CONTRI_VAL')!=0)
            .distinct()
            )
# promoad.display()
print(promoad.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
promoad_g1 = (promoad
              .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
              .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
              .filter("CONTRI_VAL = 0")
              .drop('CONTRI_VAL')
              .join((spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
                     .filter("(MODELINGNAME = 'PROMOTIONAD') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
                     .distinct()
                     ),['YEAR','MONTH','SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
               )
print(promoad_g1.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())
promoad_g2 = (promoad
              .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
              .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
              .filter("CONTRI_VAL != 0")
              .drop('CONTRI_VAL')
              .join(promoad,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'left')
              .filter(sf.col('CONTRI_VAL')!=0)
              )
print(promoad_g2.select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').distinct().count())

# COMMAND ----------

promoad_g2.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.collect_set('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()

# COMMAND ----------

print(promoad_g2.count())
print((promoad_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter(sf.col('CONTRI_VAL')<2)
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
             ).count())
print((promoad_g2
             .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                  sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                  sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
             .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
             .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA"))))
             .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
             .distinct()
             .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
          #    .filter((sf.col('CONTRI_VAL')!=1))
             .distinct()
             ).count())
print((promoad_g2
               .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                    sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                    sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
               .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA")))))
               .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
               .distinct()
               .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
               .distinct()
               ).count())

# COMMAND ----------

pa_group_one = (promoad_g2
                .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter(sf.col('CONTRI_VAL')<2)
                .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .distinct()
                .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                )
# print(pa_group_one.count())
pa_group_two = (promoad_g2
                .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA"))))
                .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                .distinct()
                .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                .filter((sf.col('CONTRI_VAL')!=1))
                .distinct()
                )
# pa_group_two.filter(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER')).display()
# pa_group_two.groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()
# pa_print(group_two.count())
w = Window.partitionBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
pa_group_three = (promoad_g2
                  .groupBy('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                       sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                       sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                  .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                  .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA")))))
                  .select('YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT')
                  .distinct()
                  .join(promoad_g2,['YEAR', 'MONTH', 'SHISHA', 'PLANNINGCUSTOMER', 'SEGMENT'],'inner')
                  .withColumn('COUNT',sf.count('YEAR').over(w))
                  .withColumn('CONTRI_VAL',sf.col('CONTRI_VAL')/sf.col('COUNT'))
                  .drop('COUNT')
                  .distinct()
                  )
pa_updt = pa_group_one.unionByName(pa_group_two).distinct().unionByName(pa_group_three).distinct()
columns = ['SPENDS', 'NETSALES', 'GPS', 'VOLUME']
for col in columns:
    pa_updt = pa_updt.withColumn(col,sf.round(sf.col(col)*sf.col('CONTRI_VAL'),3))
pa_updt = (pa_updt
           .withColumn('PLANNINGCUSTOMER_final',
                       sf.when(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER'),sf.col('PLANNINGCUSTOMER'))
                       .otherwise(sf.when((sf.col('PLANNINGCUSTOMER')!=sf.col('OLD_PLANNINGCUSTOMER'))&(sf.col('CONTRI_TYPE')=='INI_CONTRI_PO'),sf.col('PLANNINGCUSTOMER'))
                                  .otherwise(sf.col('OLD_PLANNINGCUSTOMER'))))
           .drop('PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER')
           .withColumnRenamed('PLANNINGCUSTOMER_final','PLANNINGCUSTOMER')
           .groupBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
           .agg(sf.mean('PRICE').alias('PRICE'),
                sf.sum('SPENDS').alias('SPENDS'),
                sf.sum('NETSALES').alias('NETSALES'),
                sf.sum('GPS').alias('GPS'),
                sf.sum('VOLUME').alias('VOLUME'))
           )
pa_updt.display()

# COMMAND ----------

pa_updt = pa_updt.union(promoad_g1.select(*pa_updt.columns))
print(pa_updt.count())

# COMMAND ----------

pa_updt.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
 .filter("(MODELINGNAME = 'PROMOTIONAD') and (SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0)")
 .distinct()
 ).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Combining Results

# COMMAND ----------

print(no_update.columns)
print(pa_updt.columns)
print(liq_updt.columns)
print(pa_updt.columns)

# COMMAND ----------

final_output = (no_update
                .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
                .unionByName(po_updt.withColumn('MODELINGNAME',sf.lit('PRICEOFF')))
                .unionByName(liq_updt.withColumn('MODELINGNAME',sf.lit('LIQUIDATION')))
                .unionByName(pa_updt.withColumn('MODELINGNAME',sf.lit('PROMOTIONAD')))
                .distinct()
                )
final_output.display()

# COMMAND ----------

final_output.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update))
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 ).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()

# COMMAND ----------

table_name = opt_database_name+'.'+table_to_update+'_updt'
print(table_name)
final_output.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

# COMMAND ----------

print(spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name)).columns)
print(spark.sql("select * from {0}.optimization_input_pc_updt".format(opt_database_name)).columns)

# COMMAND ----------

# opt_database_name
table_to_update

# COMMAND ----------

vw_decomp_df = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 )
vw_decomp_df.display()

# COMMAND ----------

#SPN connection interface
def getToken(sf_spn_client_id, sf_spn_client_secret):
    host = "https://login.microsoftonline.com/"
    url = "12a3af23-a769-4654-847f-958f3d479f4a/oauth2/v2.0/token"
    headers = { 'Content-Type': 'application/x-www-form-urlencoded' }
    payload = 'client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope=https://aad20912-2ec3-409d-8995-91d64894d38a/.default'.format(client_id=sf_spn_client_id,client_secret=sf_spn_client_secret)
    
    auth_token_request = requests.request("POST", host + url, headers=headers, data=payload)
    auth_response = json.loads(auth_token_request.text)
    return auth_response['access_token']
  
# Client Id of Service Principle 
spn_clientId = "f7d53643-d60f-4365-b6e4-3dc53887db73"
# Client Secret of Service Principle 
# spn_clientSecret = "OLq8Q~0iUcjfAuS~3ViI_~nj-Kn.bNHoa9KlDcWP"
# spn_clientSecret = 'oHv8Q~BZ3ul.jOYDifiTIdEuZLv9cf1EdPUfyajt'
spn_clientSecret = 'Zpu8Q~7TdcFGtprQM_MugVkdjt-jCqjSC9NP7cWe' 

auth_token = getToken(spn_clientId, spn_clientSecret)
#Access Snowflake in databricks environment to read data tables
options = {
            "sfUrl": "https://nestleprd.west-europe.privatelink.snowflakecomputing.com",
            "sfUser": "be960be3-3493-408a-b68b-cf1f9890648f",
            "sfDatabase": "JP",
            "sfAuthenticator" : "oauth",
            "sfToken" : auth_token,            
            "sfSchema": "STG",
            "sfWarehouse": "JP_MIDJP_VW1"
          }

# COMMAND ----------

# DBTITLE 1,ADDED
opti_input = spark.sql("select * from {0}.{1}".format(opt_database_name,"optimization_input"))
decomp_model_output = spark.sql(
    "select * from {0}.output_final_with_ns".format(decomp_database_name)
)
decomp_model_output = (
    decomp_model_output.withColumn(
        "STARTDATE",
        sf.to_date(
            sf.concat_ws("/", sf.lit("01"), sf.col("Month"), sf.col("Year")),
            "dd/MM/yyyy",
        ),
    )
    .filter(~sf.col("MODELINGNAME").isin(["Volume", "PREDICTED_SALES"]))
    .withColumn("HALF_YEAR", sf.when(sf.col("MONTH") <= 6, "H1").otherwise("H2"))
)

spend_list = ["PRICEOFF"]
decomp_model_output = decomp_model_output.filter(
    ~sf.col("MODELINGNAME").isin(["Volume", "PREDICTED_SALES"])
).filter(~(sf.col("MODELINGNAME").isin(spend_list)))
baseline = decomp_model_output.groupBy(
    "YEAR",
    "HALF_YEAR",
    "BUSINESS",
    "CATEGORY",
    "SUBCATEGORY",
    "CLUSTER",
    "CHANNEL",
    "SHISHA",
    "PLANNINGCUSTOMER",
    "SEGMENT",
).agg(sf.sum("Volume").alias("Baseline_Volume"))
# baseline.display()
opt_updt_final = (
    opti_input.withColumn(
        "HALF_YEAR", sf.when(sf.col("MONTH") <= 6, "H1").otherwise("H2")
    )
    .filter((sf.col("MODELINGNAME").isin(spend_list)))
    .groupBy("YEAR", "HALF_YEAR", "BUSINESS", "CATEGORY", "SUBCATEGORY", "CLUSTER", "CHANNEL", "SHISHA", "PLANNINGCUSTOMER", "SEGMENT")
    .agg(
        sf.sum("SPENDS").alias("SPENDS"),
        sf.sum("Volume").alias("VOLUME"),
        sf.sum("GPS").alias("GPS"),
        sf.sum("NETSALES").alias("NETSALES"),
        sf.avg("PRICE").alias("PRICE"),
    )
    .join(
        baseline, ["YEAR", "HALF_YEAR", "BUSINESS", "CATEGORY", "SUBCATEGORY", "CLUSTER", "CHANNEL", "SHISHA", "PLANNINGCUSTOMER", "SEGMENT"], "left"
    )
    .withColumn("TOTAL_INITIAL_SPEND", sf.col("SPENDS"))
    .withColumn("TOTAL_INITIAL_VOLUME", sf.col("VOLUME") + sf.col("Baseline_Volume"))
    .withColumn("TOTAL_INITIAL_GPS", sf.col("TOTAL_INITIAL_VOLUME") * sf.col("PRICE"))
    .withColumn(
        "TOTAL_INITIAL_NETSALES",
        sf.col("TOTAL_INITIAL_GPS") - sf.col("TOTAL_INITIAL_SPEND"),
    )
    .withColumn("MODELINGNAME", sf.lit(spend_list[0]))
    .withColumn("YEAR", sf.col("YEAR").cast("int") + 1)
    .withColumn("IS_LATEST", sf.lit(1))
    .fillna(
        0,
        subset=[
            "TOTAL_INITIAL_SPEND",
            "TOTAL_INITIAL_VOLUME",
            "TOTAL_INITIAL_GPS",
            "TOTAL_INITIAL_NETSALES",
        ],
    )
)
opt_updt_final = opt_updt_final.withColumn(
    "MONTH",
    sf.when(sf.col("HALF_YEAR") == "H1", sf.lit(6)).when(
        sf.col("HALF_YEAR") == "H2", sf.lit(12)
    ),
)

# Construct end_of_period
opt_updt_final = opt_updt_final.withColumn(
    "end_of_period",
    sf.last_day(
        sf.to_date(sf.concat_ws("-", sf.col("YEAR"), sf.col("MONTH"), sf.lit("01")))
    ),
)

# Add time 23:59:59
opt_updt_final = opt_updt_final.withColumn(
    "DATE",
    sf.concat_ws(" ", sf.col("end_of_period").cast("string"), sf.lit("23:59:59")),
)

spend_list_all = [["PROMOTIONAD"], ["LIQUIDATION"]]
final_historical = opt_updt_final
for spend_list in spend_list_all:
    decomp_model_output = decomp_model_output.filter(
        ~sf.col("MODELINGNAME").isin(["Volume", "PREDICTED_SALES"])
    ).filter(~(sf.col("MODELINGNAME").isin(spend_list)))
    baseline = decomp_model_output.groupBy(
        "YEAR",
        "HALF_YEAR",
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "CLUSTER",
        "CHANNEL",
        "SHISHA",
        "PLANNINGCUSTOMER",
        "SEGMENT",
    ).agg(sf.sum("Volume").alias("Baseline_Volume"))
    # baseline.display()
    opt_updt_final = (
        opti_input.withColumn(
            "HALF_YEAR", sf.when(sf.col("MONTH") <= 6, "H1").otherwise("H2")
        )
        .filter((sf.col("MODELINGNAME").isin(spend_list)))
        .groupBy(
            "YEAR",
            "HALF_YEAR",
            "BUSINESS",
            "CATEGORY",
            "SUBCATEGORY",
            "CLUSTER",
            "CHANNEL",
            "SHISHA",
            "PLANNINGCUSTOMER",
            "SEGMENT",
        )
        .agg(
            sf.sum("SPENDS").alias("SPENDS"),
            sf.sum("Volume").alias("VOLUME"),
            sf.sum("GPS").alias("GPS"),
            sf.sum("NETSALES").alias("NETSALES"),
            sf.avg("PRICE").alias("PRICE"),
        )
        .join(
            baseline,
            ["YEAR", "HALF_YEAR", "BUSINESS", "CATEGORY", "SUBCATEGORY", "CLUSTER", "CHANNEL", "SHISHA", "PLANNINGCUSTOMER", "SEGMENT"],
            "left",
        )
        .withColumn("TOTAL_INITIAL_SPEND", sf.col("SPENDS"))
        .withColumn(
            "TOTAL_INITIAL_VOLUME", sf.col("VOLUME") + sf.col("Baseline_Volume")
        )
        .withColumn(
            "TOTAL_INITIAL_GPS", sf.col("TOTAL_INITIAL_VOLUME") * sf.col("PRICE")
        )
        .withColumn(
            "TOTAL_INITIAL_NETSALES",
            sf.col("TOTAL_INITIAL_GPS") - sf.col("TOTAL_INITIAL_SPEND"),
        )
        .withColumn("MODELINGNAME", sf.lit(spend_list[0]))
        .withColumn("YEAR", sf.col("YEAR").cast("int") + 1)
        .withColumn("IS_LATEST", sf.lit(1))
        .fillna(
            0,
            subset=[
                "TOTAL_INITIAL_SPEND",
                "TOTAL_INITIAL_VOLUME",
                "TOTAL_INITIAL_GPS",
                "TOTAL_INITIAL_NETSALES",
            ],
        )
    )
    opt_updt_final = opt_updt_final.withColumn(
        "MONTH",
        sf.when(sf.col("HALF_YEAR") == "H1", sf.lit(6)).when(
            sf.col("HALF_YEAR") == "H2", sf.lit(12)
        ),
    )

    # Construct end_of_period
    opt_updt_final = opt_updt_final.withColumn(
        "end_of_period",
        sf.last_day(
            sf.to_date(sf.concat_ws("-", sf.col("YEAR"), sf.col("MONTH"), sf.lit("01")))
        ),
    )

    # Add time 23:59:59
    opt_updt_final = opt_updt_final.withColumn(
        "DATE",
        sf.concat_ws(" ", sf.col("end_of_period").cast("string"), sf.lit("23:59:59")),
    )
    final_historical = final_historical.union(opt_updt_final)

final_historical = (
    final_historical.withColumn("SCENARIO_NAME", sf.lit("HISTORICAL_DATA"))
    .select(
        "SCENARIO_NAME",
        "YEAR",
        "HALF_YEAR",
        "DATE",
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "CLUSTER",
        "CHANNEL",
        "SHISHA",
        "PLANNINGCUSTOMER",
        "SEGMENT",
        "MODELINGNAME",
        "BASELINE_VOLUME",
        "PRICE",
        "TOTAL_INITIAL_SPEND",
        "TOTAL_INITIAL_VOLUME",
        "TOTAL_INITIAL_GPS",
        "TOTAL_INITIAL_NETSALES",
        "IS_LATEST",
    )
    .distinct()
)

mid_jp = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT")
    .load()
    .filter(sf.col("SCENARIO_NAME") != "HISTORICAL_DATA")
    .withColumn(
        "IS_LATEST", sf.lit(0)
    )
)
for col in list(set(mid_jp.columns) - set(final_historical.columns)):
    final_historical = final_historical.withColumn(col, sf.lit(None))
mid_jp_updt = mid_jp.unionByName(final_historical, allowMissingColumns=True)
mid_jp_updt.filter("SCENARIO_NAME == 'HISTORICAL_DATA'").display()

mid_jp_updt.write.format("snowflake").options(**options).option(
    "dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT"
).mode("overwrite").save()
print("Data loaded in backend table")

# COMMAND ----------

mid_jp_updt.groupBy("MODELINGNAME").agg(sf.count('*')).display()

# COMMAND ----------

vw_decomp_df = spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
vw_decomp_df.display()
vw_decomp_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable("vw_decomp_df_20jun25")

# COMMAND ----------

# MAGIC %md # Check

# COMMAND ----------

vw_decomp_df.agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME"), sf.count("*").alias("count of rows")).display()

# COMMAND ----------

# DBTITLE 1,(testing)
agg_df = vw_decomp_df.agg(
    sf.sum("SPENDS").alias("SPENDS"),
    sf.sum("GPS").alias("GPS"),
    sf.sum("NETSALES").alias("NETSALES"),
    sf.sum("VOLUME").alias("VOLUME"),
    sf.count("*").alias("count_of_rows")
)

# Format large numbers into decimal strings (2 decimal places shown here — adjust as needed)
formatted_df = agg_df.select(
    sf.format_number("SPENDS", 2).alias("SPENDS"),
    sf.format_number("GPS", 2).alias("GPS"),
    sf.format_number("NETSALES", 2).alias("NETSALES"),
    sf.format_number("VOLUME", 2).alias("VOLUME"),
    "count_of_rows"
)

# Display the formatted output
formatted_df.display()

# COMMAND ----------


vw_decomp_df.filter(sf.col('PLANNINGCUSTOMER') == 'L5 JP NS Kanesue Honbu').display()

# COMMAND ----------

#old one
vw_decomp_df.filter(sf.col('PLANNINGCUSTOMER') == 'L5 JP NS Kanesue Honbu').display()

# COMMAND ----------

# DBTITLE 1,check after tech team has updated vw_decomp

#SPN connection interface
def getToken(sf_spn_client_id, sf_spn_client_secret):
    host = "https://login.microsoftonline.com/"
    url = "12a3af23-a769-4654-847f-958f3d479f4a/oauth2/v2.0/token"
    headers = { 'Content-Type': 'application/x-www-form-urlencoded' }
    payload = 'client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope=https://aad20912-2ec3-409d-8995-91d64894d38a/.default'.format(client_id=sf_spn_client_id,client_secret=sf_spn_client_secret)
    
    auth_token_request = requests.request("POST", host + url, headers=headers, data=payload)
    auth_response = json.loads(auth_token_request.text)
    return auth_response['access_token']
  
# Client Id of Service Principle 
spn_clientId = 'f85542e4-f419-49ec-9c29-ad875e32dde1'
# Client Secret of Service Principle 
# spn_clientSecret = 'Pb78Q~NyxLU6TFe.ZCiD8rJeTICTCCg1AZgocbj2' 
spn_clientSecret = 'oHv8Q~BZ3ul.jOYDifiTIdEuZLv9cf1EdPUfyajt' 

auth_token = getToken(spn_clientId, spn_clientSecret)
#Access Snowflake in databricks environment to read data tables
options = {
            "sfUrl": "https://nestleprd.west-europe.privatelink.snowflakecomputing.com",
            "sfUser": "390877d5-c339-40a3-99ad-8d4ffe306c14",
            "sfDatabase": "JP",
            "sfAuthenticator" : "oauth",
            "sfToken" : auth_token,            
            "sfSchema": "PRS",
            "sfWarehouse": "JP_MIDJP_VW1"
          }

decomp_output = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "VW_DECOMP_OUTPUT")
    .load()
)
decomp_output.agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME"), sf.count("*").alias("count of rows")).display()

# COMMAND ----------

decomp_output.filter(sf.col("PLANNINGCUSTOMER")=='L5 JP NS Kanesue Honbu').display()

# COMMAND ----------

missing_cust=["L5 JP NS Sugiyama Yakuhin Honbu",
"L5 JP NS Fuji Honbu",
"L5 JP NS Aoki Super Honbu",
"L5 JP NS Group Plan CGC First CHUBU",
"L5 JP NS Super Sanshi",
"L5 JP NS FEEL",
"L5 JP NS Toyota COOP Shohinbu",
"L5 JP NS AEON RETAIL",
"L5 JP NS Domy Honbu",
"L5 JP NS Fuji Maxvale Yago Honbu",
"L5 JP NS FUJI The BIG Yago"
]
vw_decomp_df.groupby("PLANNINGCUSTOMER").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES")).filter(sf.col("PLANNINGCUSTOMER").isin(missing_cust)).display()

# COMMAND ----------

customer=['L5 JP NS Lopia Kansai Area','L5 JP NS Tokou Store Honbu','L5 JP NS Lopia Kyushu Area','L5 JP NS Doutou Arcs Honbu','L5 JP NS TEURI CNF KANSAI AREA','L5 JP NS Kanesue Honbu','L5 JP NS Lopia Chubu Area','L5 JP NS Donan RALSE Honbu','L5 JP NS Big Fuji Honbu']
vw_decomp_df.groupby("PLANNINGCUSTOMER").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES")).filter(sf.col("PLANNINGCUSTOMER").isin(customer)).display()

# COMMAND ----------

decomp_output = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "VW_DECOMP_OUTPUT")
    .load()
)
decomp_output.filter("MODELINGNAME NOT IN ('PREDICTED_SALES', 'Volume')").withColumn("YEARMONTH", sf.concat(sf.col("YEAR"), sf.lpad(sf.col("MONTH"), 2, "0"))).groupBy("YEARMONTH").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME"), sf.count("*").alias("count of rows")).display()

# COMMAND ----------

decomp_output.select("YEAR", "MONTH").distinct().orderBy("YEAR", "MONTH").display()

# COMMAND ----------

check = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
#  .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .filter("SPENDS > 0 or VOLUME >= 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 )

# COMMAND ----------

check.dtypes

# COMMAND ----------

check.display()

# COMMAND ----------

check.agg(sf.sum('SPENDS')).display()




(spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).agg(sf.sum('SPENDS')).display()


check_raw = check.filter(~sf.col('MODELINGNAME').isin('PREDICTED_SALES','Volume')).filter(sf.col("DATE")>= '2024-07-01 00:00:00').filter(sf.col("DATE")<= '2024-12-31 23:59:59').filter(sf.col("SPENDS")> 0).filter(sf.col("VOLUME")>= 0).filter(sf.col("BUSINESS").isin('JP/Coffee&Beverage')).filter(sf.col("CATEGORY").isin('JP/C&B/RTD')).filter(sf.col("MODELINGNAME").isin('PRICEOFF')).distinct()
#   .filter(1=1 AND 1=1 AND "MODELINGNAME" NOT IN ('PREDICTED_SALES','Volume') AND "DATE" >= '2025-07-01 00:00:00' AND "DATE" <= '2025-12-31 23:59:59' AND "BUSINESS" = 'JP/Coffee&Beverage' AND "CATEGORY" = 'JP/C&B/RTD' AND "MODELINGNAME" = 'PRICEOFF' AND "SCENARIO_NAME" = '07_05_RTD_ALL_H2') GROUP BY PLANNINGCUSTOMER
# .distinct()
        #   )
# check_decomp.display()

# check.groupBy('SHISHA').agg(sf.sum('SPENDS')).display()
check_raw.groupBy('PLANNINGCUSTOMER').agg(sf.sum('SPENDS')).display()




# check.display()

check_decomp = (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()
        #   .filter(sf.col('SCENARIO_NAME').contains(SCENARIONAME))
          .filter(~sf.col('MODELINGNAME').isin('PREDICTED_SALES','Volume'))
          .filter(sf.col("DATE")>= '2024-07-01 00:00:00')
          .filter(sf.col("DATE")<= '2024-12-31 23:59:59')
          .filter(sf.col("SPENDS")> 0)
          .filter(sf.col("VOLUME")>= 0)
          .filter(sf.col("BUSINESS").isin('JP/Coffee&Beverage'))
          .filter(sf.col("CATEGORY").isin('JP/C&B/RTD'))
          .filter(sf.col("MODELINGNAME").isin('PRICEOFF'))
        #   .filter(1=1 AND 1=1 AND "MODELINGNAME" NOT IN ('PREDICTED_SALES','Volume') AND "DATE" >= '2025-07-01 00:00:00' AND "DATE" <= '2025-12-31 23:59:59' AND "BUSINESS" = 'JP/Coffee&Beverage' AND "CATEGORY" = 'JP/C&B/RTD' AND "MODELINGNAME" = 'PRICEOFF' AND "SCENARIO_NAME" = '07_05_RTD_ALL_H2') GROUP BY PLANNINGCUSTOMER
          .distinct()
          )
# check_decomp.display()

# check.groupBy('SHISHA').agg(sf.sum('SPENDS')).display()
check_decomp.groupBy('PLANNINGCUSTOMER').agg(sf.sum('SPENDS')).display()

# COMMAND ----------



# COMMAND ----------

# check.groupBy('PLANNINGCUSTOMER').agg(sf.sum('SPENDS')).display()
check.filter(sf.col("DATE")>= '2024-07-01 00:00:00').filter(sf.col("DATE")<= '2024-12-31 23:59:59').filter(sf.col("SPENDS")> 0).filter(sf.col("VOLUME")>= 0).filter(sf.col("BUSINESS").isin('JP/Coffee&Beverage')).filter(sf.col("CATEGORY").isin('JP/C&B/RTD')).filter(sf.col("MODELINGNAME").isin('PRICEOFF')).groupBy('PLANNINGCUSTOMER').agg(sf.sum('SPENDS')).display()
# check.display()

# COMMAND ----------

check.filter("MODELINGNAME  not in ('Volume','PREDICTED_SALES')").filter(sf.col("CATEGORY")=="JP/CNF/In To Home").groupBy("YEAR","MONTH").agg(sf.sum('NETSALES')).display()

# COMMAND ----------

# (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(sf.col("YEAR").isin(2024)).display()

(spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter("MODELINGNAME  not in ('Volume','PREDICTED_SALES')").filter(sf.col("CATEGORY")=="JP/CNF/In To Home").groupBy("YEAR","MONTH").agg(sf.sum('NETSALES')).display()

# COMMAND ----------

check.filter("MODELINGNAME  not in ('Volume','PREDICTED_SALES')").groupBy("YEAR").agg(sf.sum('SPENDS'),sf.sum('NETSALES'),sf.sum('GPS')).display()

# COMMAND ----------

final_output.display()

# COMMAND ----------

final_output.select("MODELINGNAME").distinct().display()

database_name = "tpo_output_v18"

opt_database_name = "ui_output_v13"

check = (spark.sql("select * from {0}.output_final_with_ns".format(database_name)))
check.select("MODELINGNAME").distinct().display()

check_1 = (spark.sql("select * from {0}.optimization_input_pc_updt".format(opt_database_name)))
check_1.select("MODELINGNAME").distinct().display()

# print(spark.sql("select * from {0}.optimization_input_pc_updt".format(opt_database_name)).columns)

# check = (spark.sql("select * from {0}.mbt_wo_competition_price_corrected".format(database_name)))
# check.select("MODELINGNAME").distinct().display()

# check.display()

# (spark.sql("select * from {0}.output_final_with_ns".format(database_name))
#  .filter("MODELINGNAME  not in ('Volume','PREDICTED_SALES','Base_BasePrice')")
#  .select(sf.sum('SPENDS'),sf.sum('NETSALES'),sf.sum('GPS'))
#  ).display()

# COMMAND ----------



# COMMAND ----------

# check.display()
check.filter(sf.col("YEAR").isin(2023,2024)).display()


# COMMAND ----------

(spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(sf.col("YEAR").isin(2023,2024)).display()

# COMMAND ----------

check_base = check

check_0 = (
    check_base.filter(~sf.col("MODELINGNAME").isin("Volume", "PREDICTED_SALES"))
    .groupBy("YEAR", 
             "MONTH", 
             "SEGMENT", 
            #  "SHISHA", 
            #  "PLANNINGCUSTOMER"
             )
    .agg(sf.sum("SPENDS"), sf.sum("GPS"), sf.sum("NETSALES"))
)
check_0.display()

# COMMAND ----------

check_base = check

check_0 = check_base.filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).groupBy("YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER").agg(sf.sum("SPENDS"),sf.sum("GPS"),sf.sum("NETSALES"))
check_0.display()


database_name = "tpo_output_v17"
check = (spark.sql("select * from {0}.mbt_wo_competition_price_corrected".format(database_name)))
# check.display()
check = check.withColumnRenamed("SHISHANM","SHISHA").withColumnRenamed("SALESAMT","GPS").withColumn("SPENDS",sf.col("PRICEOFF")+sf.col("LIQUIDATION")+sf.col("PROMOTIONAD")).withColumn("NETSALES",sf.col("GPS")-sf.col("SPENDS"))
check = check.select("YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER","SPENDS","GPS","NETSALES")
check = check.groupBy("YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER").agg(sf.sum("SPENDS").alias("SPENDS"),sf.sum("GPS").alias("GPS"),sf.sum("NETSALES").alias("NETSALES"))
# check.display()


check_final = check_0.join(check,["YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER"],how="left")
check_final = check_final.withColumn("mult_SPENDS",sf.col("SPENDS")/sf.col("sum(SPENDS)")).withColumn("mult_GPS",sf.col("GPS")/sf.col("sum(GPS)")).withColumn("mult_NETSALES",sf.col("NETSALES")/sf.col("sum(NETSALES)"))

check_final = check_final.select("YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER","mult_NETSALES","mult_GPS","mult_SPENDS")
check_final = check_final.fillna(1)
# check_final.display()


final_df_after_imp = check_base.join(check_final,["YEAR","MONTH","SEGMENT","SHISHA","PLANNINGCUSTOMER"],how="left")
final_df_after_imp = final_df_after_imp.withColumn("final_SPENDS",sf.col("mult_SPENDS")*sf.col("SPENDS")).withColumn("final_GPS",sf.col("mult_GPS")*sf.col("GPS")).withColumn("final_NETSALES",sf.col("mult_NETSALES")*sf.col("NETSALES"))
final_df_after_imp = final_df_after_imp.drop("mult_SPENDS","mult_GPS","mult_NETSALES","SPENDS","GPS","NETSALES")
final_df_after_imp = final_df_after_imp.withColumnRenamed("final_SPENDS","SPENDS").withColumnRenamed("final_GPS","GPS").withColumnRenamed("final_NETSALES","NETSALES")
# final_df_after_imp.display()


final_df_after_imp.filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").groupBy("YEAR","MONTH").agg(sf.sum("NETSALES")).display()

final_df_after_imp.filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").groupBy("YEAR","MONTH","SEGMENT").agg(sf.sum("NETSALES")).display()

# COMMAND ----------

final_df_after_imp = final_df_after_imp.fillna(0)
final_df_after_imp.display()

# check.filter(sf.col("YEAR")==2024).filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).groupBy("YEAR","MONTH","CATEGORY","SUBCATEGORY","SEGMENT","SHISHA","CLUSTER","CHANNEL","PLANNINGCUSTOMER","MODELINGNAME").agg(sf.sum("SPENDS"),sf.sum("NETSALES"),sf.sum("GPS")).display()

# (spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).filter(sf.col("YEAR")==2024).filter(~sf.col("MODELINGNAME").isin("Volume","PREDICTED_SALES")).groupBy("YEAR","MONTH","CATEGORY","SUBCATEGORY","SEGMENT","SHISHA","CLUSTER","CHANNEL","PLANNINGCUSTOMER","MODELINGNAME").agg(sf.sum("SPENDS"),sf.sum("NETSALES"),sf.sum("GPS")).display()

# database_name = "tpo_output_v16"
# mbt_data = (spark.sql("select * from {0}.mbt_wo_competition_price_corrected".format(database_name)))
# mbt_data.display()

# COMMAND ----------

check.columns

# COMMAND ----------

# check.groupBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT').agg(sf.sum("SPENDS"),sf.sum("NETSALES")).display()

# (spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).groupBy('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT').agg(sf.sum("SPENDS"),sf.sum("NETSALES")).display()
# database_name = "tpo_output_v16"
# sellout_data = (spark.sql("select * from {0}.sellout_by_day_view".format(database_name))
#                 .withColumn('ACTUALDTM',sf.to_date(sf.col('ACTUALDTM'),"yyyyMMdd"))
#                 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
#                 .withColumn('YEARMONTH', sf.concat(sf.col('YEAR'),sf.col('MONTH')))
#             #    .filter((sf.col('YEARMONTH')>=start_yearmonth) & (sf.col('YEARMONTH')<=end_yearmonth))
#                )
# # print(spends_data.count())
# sellout_data.select("SEGMENT").distinct().display()

# check.select("MODELINGNAME").distinct().display()

# (spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).select("MODELINGNAME").distinct().display()

(check).select("MODELINGNAME").distinct().display()

(spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).select("MODELINGNAME").distinct().display()

# (spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).groupBy("YEAR","MONTH","SEGMENT").agg(sf.sum("SPENDS"),sf.sum("NETSALES"),sf.sum("GPS")).display()

# COMMAND ----------

check.filter(~sf.col("MODELINGNAME").isin("PREDICTED_SALES","Volume")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").filter(sf.col("YEAR")==2024).filter(sf.col("MONTH")==2).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
# check.select("CATEGORY").distinct().display()
# check.select("MODELINGNAME").distinct().display()
# check.filter(~sf.col("MODELINGNAME").isin("PREDICTED_SALES","Volume")).select("MODELINGNAME").distinct().display()

# COMMAND ----------

print((spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).filter(~sf.col("MODELINGNAME").isin("PREDICTED_SALES","Volume")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").filter(sf.col("YEAR")==2024).filter(sf.col("MONTH")==2).select(sf.sum("NETSALES")).display())

print((spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(~sf.col("MODELINGNAME").isin("PREDICTED_SALES","Volume")).filter(sf.col("CATEGORY")=="JP/CNF/In To Home").filter(sf.col("YEAR")==2024).filter(sf.col("MONTH")==2).select(sf.sum("NETSALES")).display())

# COMMAND ----------

# # check.select('DATE').distinct().display()
# print(check.filter(sf.col("YEAR")>=2022).groupBy('YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).count())

# check.filter(sf.col("YEAR")>=2022).groupBy('YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()

# check.groupBy('SEGMENT').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()

check.groupBy('DATE',"SEGMENT").agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()
# check.filter(sf.col("DATE")<"2024-09-31").filter(sf.col("DATE")>="2021-11-30").groupBy('DATE',"SEGMENT").agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()
# check.filter(sf.col("DATE")<="2024-09-01").groupBy('DATE').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()

# COMMAND ----------

(spark.sql("select * from vw_decomp_df_20jun25")).display()

# COMMAND ----------

print((spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).select(sf.sum("NETSALES")).display())

table_name = "vw_decomp_output_nov_2024_refresh_data_till_aug_2024"

(spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

print((spark.sql("select * from vw_decomp_output_nov_2024_refresh_data_till_aug_2024")).select(sf.sum("NETSALES")).display())


# COMMAND ----------

# spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt')
table_name = 'ui_output_v9'+'.'+'vw_decomp_output'+'_backup_sep_refresh'
table_name
# (spark.sql("select * from table_name")).load().count()
# (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).count()
# print((spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(sf.col("YEAR")>=2022).count())
# print((spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(sf.col("YEAR")>=2022).groupBy('YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).count())

# (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).filter(sf.col("YEAR")>=2022).groupBy('YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME').agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()

# (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).display()

# (spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).select("DATE").distinct().display()

(spark.read.format("snowflake").options(**options).option("dbtable","VW_DECOMP_OUTPUT").load()).groupBy('DATE',"SEGMENT").agg(sf.sum('SPENDS'), sf.sum('GPS'), sf.sum('NETSALES'), sf.sum('VOLUME')).display()

# COMMAND ----------



# COMMAND ----------

(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 ).filter(sf.col("MODELINGNAME")!="PREDICTED_SALES").filter(sf.col("MODELINGNAME")!="Volume").filter(sf.col("YEAR") == 2023).groupBy("SEGMENT").agg(sf.sum("NETSALES").alias('NETSALES'),sf.sum("SPENDS").alias('SPENDS')).display()

# COMMAND ----------

(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 ).filter(sf.col("MODELINGNAME")!="PREDICTED_SALES").filter(sf.col("SEGMENT")=="JP/CNF/In To Home/KKCORE/M-Bag").filter(sf.col("MODELINGNAME")!="Volume").display()

# COMMAND ----------

check_del = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 )

# check_del.select('MODELINGNAME').distinct().collect()
check_del.filter(sf.col("MODELINGNAME")!="PREDICTED_SALES").filter(sf.col("MODELINGNAME")!="Volume").filter(sf.col("YEAR") == 2023).groupBy("SEGMENT").agg(sf.sum("NETSALES").alias('NETSALES')).display()
check_del.display()


# COMMAND ----------

print(check_del.count())
check_del = check_del.filter(sf.col("CATEGORY").isin("JP/CNF/In To Home"))
print(check_del.count()) 

# COMMAND ----------

check_output = (
    check_del.where(sf.col("DATE") >= "2024-01-01 00:00:00")
    .where(sf.col("DATE") <= "2024-06-30 23:59:59")
    .where(sf.col("MODELINGNAME") == "PRICEOFF")
    .where(sf.col("BUSINESS") == "JP/Confectionery")
    .where(sf.col("CATEGORY") == "JP/CNF/In To Home")
    .where(sf.col("SPENDS") > 0)
    .where(sf.col("VOLUME") >= 0)
    .groupBy("SEGMENT")
    .agg(sf.sum("NETSALES").alias("NETSALES"))
)
check_output.display()

# COMMAND ----------

a = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 ).display()


# COMMAND ----------

a = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 )

# COMMAND ----------

a.filter(sf.col("MODELINGNAME")=="Volume").display()

# COMMAND ----------

a.groupby('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER').agg(sf.sum('SPENDS'),sf.sum('GPS'),sf.sum('NETSALES'),sf.sum('VOLUME')).display()

# COMMAND ----------

a.select(sf.col("SEGMENT")).distinct().display()

# COMMAND ----------

decomp = (spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct()
 )

# COMMAND ----------

add_new_data.display()
add_new_data_vol.display()
add_new_data_PS.display()

# COMMAND ----------

print(add_new_data.columns)
print(add_new_data_PS.columns)
print(add_new_data_Vol.columns)

# COMMAND ----------

# actual_data.display()

# actual_data.filter(sf.col("YEAR") == 2023).filter(sf.col("MONTH") == 12).filter(sf.col("SEGMENT").contains("JP/CNF/In To Home")).display()
# .select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
# .filter(
#     sf.col("CATEGORY") == "JP/CNF/In To Home")
    # .display()

# new_data.filter(sf.col("YEAR") == 2023).filter(sf.col("MONTH") == 12).filter(sf.col("SEGMENT").contains("JP/CNF/In To Home")).select("SEGMENT").distinct().display()

# new_data.filter(sf.col("YEAR") == 2023).filter(sf.col("MONTH") == 12).filter(sf.col("SEGMENT").contains("JP/CNF/In To Home/KKLE HR/M-bag")).select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()

new_data.filter(sf.col("SEGMENT").contains("JP/CNF/In To Home")).display()

add_new_data = new_data.filter(sf.col("SEGMENT").isin(["JP/CNF/In To Home/KKLE HR/M-bag",
                                                       "JP/CNF/In To Home/KKLE HR/S-bag",
                                                       "JP/CNF/In To Home/KKCORE/M-Bag",
                                              "JP/CNF/In To Home/KKCORE/S-Bag"]))
groupkey = ['YEAR','MONTH','BUSINESS','CATEGORY','SUBCATEGORY','SEGMENT','SHISHA','CLUSTER','CHANNEL','PLANNINGCUSTOMER']
add_new_data_tot = add_new_data.groupBy(groupkey).agg(sf.sum("SPENDS").alias("SPENDS"),
                                                      sf.sum("GPS").alias("GPS"),
                                                      sf.sum("NETSALES").alias("NETSALES"),
                                                      sf.sum("VOLUME").alias("VOLUME"),
                                                      sf.avg("PRICE").alias("PRICE"),
                                                      sf.avg("EC").alias("EC"),
                                                      sf.avg("EC_PER_QTYCASE").alias("EC_PER_QTYCASE"),
                                                      sf.avg("MC").alias("MC"))
add_new_data_PS = add_new_data_tot.withColumn("MODELINGNAME", sf.lit("PREDICTED_SALES"))

add_new_data_Vol = add_new_data_tot.withColumn("MODELINGNAME", sf.lit("Volume"))

add_new_data_PS = add_new_data_PS.select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME', 'EC', 'MC', 'PRICE', 'EC_PER_QTYCASE')

add_new_data_Vol = add_new_data_Vol.select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME', 'EC', 'MC', 'PRICE', 'EC_PER_QTYCASE')

final_add_new_data = add_new_data.union(add_new_data_PS).union(add_new_data_Vol)

final_add_new_data = (final_add_new_data.withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
.withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2))))

for col_name in ['SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC']:
    final_add_new_data = final_add_new_data.withColumnRenamed(col_name, "{}_new".format(col_name))
join_cols = ['DATE','YEAR','MONTH','HALF_YEAR','BUSINESS','CATEGORY','SUBCATEGORY','SEGMENT','SHISHA','CLUSTER','CHANNEL','PLANNINGCUSTOMER','MODELINGNAME']
decomp = decomp.join(final_add_new_data, on=join_cols, how="outer")

for col_name in ['SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC']:
    decomp = decomp.withColumn(col_name,sf.when(sf.col("SEGMENT").isin(["JP/CNF/In To Home/KKLE HR/M-bag",
                                                       "JP/CNF/In To Home/KKLE HR/S-bag",
                                                       "JP/CNF/In To Home/KKCORE/M-Bag",
                                                       "JP/CNF/In To Home/KKCORE/S-Bag"])&(sf.col("MODELINGNAME")!="PREDICTED_SALES"), 
                                                sf.col("{}_new".format(col_name)))
                               .otherwise(sf.col(col_name))
                                )
decomp.display()
# groupby by removing volumn and predicted sales
# you well get vol
# join again 

for col_name in ['SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC']:
    decomp = decomp.withColumn(col_name,sf.when(sf.col("SEGMENT").isin(["JP/CNF/In To Home/KKLE HR/M-bag",
                                                       "JP/CNF/In To Home/KKLE HR/S-bag",
                                                       "JP/CNF/In To Home/KKCORE/M-Bag",
                                                       "JP/CNF/In To Home/KKCORE/S-Bag"])&
                                                (sf.col("MODELINGNAME")=="PREDICTED_SALES")&
                                                (sf.col(col_name).isNull()), 
                                                sf.col("{}_new".format(col_name)))
                               .otherwise(sf.col(col_name))
                                )                                      

decomp = decomp.drop(*["{}_new".format(col_name) for col_name in ['SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC']])

# JP/CNF/In To Home/KKLE HR/M-bag
# JP/CNF/In To Home/KKLE HR/S-bag

# JP/CNF/In To Home/KKCORE/M-Bag
# JP/CNF/In To Home/KKCORE/S-Bag



# COMMAND ----------

# SCENARIONAME = "26_11_24_RSC_ALL"


data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load()
# .filter(sf.col('SCENARIONAME')!=scenario_name)

# input_ = (spark.read.format("snowflake").options(**options).option("dbtable","MODELPARAMETERS").load()
#         #   .filter(sf.col('PARAMETERS').contains(SCENARIONAME))
#           .distinct()
#           )
# input_.display()

# COMMAND ----------

# check = decomp.filter(sf.col("SEGMENT").isin(["JP/CNF/In To Home/KKLE HR/M-bag",
#                                                        "JP/CNF/In To Home/KKLE HR/S-bag",
#                                                        "JP/CNF/In To Home/KKCORE/M-Bag",
#                                               "JP/CNF/In To Home/KKCORE/S-Bag"]))
# check.display()

print(decomp.count())
decomp.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
check = decomp
# check.select([sf.count(sf.when(sf.isnan(c), c)).alias(c) for c in check.columns]).show()
Dict_Null = {col:check.filter(check[col].isNull()).count() for col in check.columns}
print(Dict_Null)
check = check.na.fill(value=0)
Dict_Null = {col:check.filter(check[col].isNull()).count() for col in check.columns}
print(Dict_Null)
check.select(sf.sum('SPENDS'),sf.sum('VOLUME'),sf.sum('GPS'),sf.sum('NETSALES')).display()
# df.na.fill(0).show()

# print(check.count())
check.display()

# COMMAND ----------

()(spark.sql("select * from {0}.{1}".format(opt_database_name,table_to_update+'_updt'))
 .unionByName((spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name))
               .filter("MODELINGNAME in ('Volume','PREDICTED_SALES')")
               .select('YEAR', 'MONTH', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'PRICE', 'SPENDS', 'NETSALES', 'GPS', 'VOLUME')
               .distinct()
               ))
 .withColumn('DATE',sf.last_day(sf.concat_ws("-",sf.col('YEAR'),sf.col('MONTH'),sf.lit('01'))))
 .withColumn('DATE',sf.concat_ws(" ", sf.col('DATE'), sf.lit('23:59:59')))
 .withColumn('MONTH',sf.col('MONTH').cast('string'))
 .withColumn('YEAR',sf.col('YEAR').cast('string'))
 .withColumn('MONTH',sf.when(sf.length(sf.col('MONTH'))==1,sf.concat(sf.lit('0'),sf.col('MONTH'))).otherwise(sf.col('MONTH')))
 .withColumn("HALF_YEAR",sf.when(sf.col('MONTH').isin(['01','02','03','04','05','06']),sf.lit(1)).otherwise(sf.lit(2)))
 .withColumn('EC',sf.lit(0))
 .withColumn('EC_PER_QTYCASE',sf.lit(0))
 .withColumn('MC',sf.lit(0))
 .select('DATE', 'YEAR', 'MONTH', 'HALF_YEAR', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'SPENDS', 'GPS', 'NETSALES', 'VOLUME', 'PRICE', 'EC', 'EC_PER_QTYCASE', 'MC')
 .filter("SPENDS != 0 or VOLUME != 0 or GPS != 0 or NETSALES != 0")
 .distinct().filter(sf.col("SEGMENT").contains("JP/CNF/In To Home"))
 ).display()

# COMMAND ----------

# cat_name = 'PBC_JP_1_ARCH'
# cat_name = 'RSCIC_JP_1_ARCH' 
# cat_name = 'MIXES_JP_1_ARCH'
cat_name = 'CHOCO_JP_1_ARCH'
df = spark.read.csv("{}/category/{}/data_prep.csv".format(base_path_spark, cat_name), header=True)
df.display()

# COMMAND ----------

