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
import scipy.optimize as ot
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
from pandas.io import sql
import time
from multiprocessing import Process, Queue, Pool, set_start_method
import os
# import ast
import concurrent.futures
import requests, json, logging, sys
from pyspark.sql.functions import concat
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

# decomp_df_base = (
#     spark.read.format("snowflake")
#     .options(**options)
#     .option("dbtable", "VW_DECOMP_OUTPUT")
#     .load()
# )
# decomp_df_base.display()

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

decomp_df_base = (spark.sql("select * from vw_decomp_df_20jun25"))
decomp_df_base.display()

# COMMAND ----------

uid = sys.argv[1]
# uid = '{"uid":"uideopbfZYKId_2024_11_25_08_06_29"}'
# uid = '{"uid":"uideVmrMXfrKy_2025_06_16_05_55_57"}'
# uid = '{"uid":"uidshBu4zmbbR_2025_06_18_08_12_53"}'
# uid = '{"uid":"uidWutTZYqoe7_2025_07_15_07_45_39"}'


# ast.literal_eval(uid)['uid']
uid = json.loads(uid)['uid']
print(uid)
input_ = (spark.read.format("snowflake").options(**options).option("dbtable","MODELPARAMETERS").load()
          .filter(sf.col('UID')==uid)
          .select('UID','PARAMETERS')
          .distinct()
          ).toPandas()
input_params = input_.loc[0,'PARAMETERS']
input_params = json.loads(input_params)

# input_params = {"scenarioPrameters": {"scenarioName": "scenario2"},"optimizationInput": {"business": "JP/Confectionery", "spendtype": "PRICEOFF", "channel": ["SM"], "category": ["JP/CNF/In To Home"], "budget": 0, "optimizationobjective": "Maximize revenue keeping spends same as last quarter", "max": "2024-06-30 23:59:59", "min": "2024-01-01 00:00:00", "shisha": ["TOKYO", "KITANIHON", "KYUSHU", "CHUBU", "CHUSHIKOKU", "KANSAI", "NRF", "NRS"]}}
print(input_params)

# COMMAND ----------

# CNF test budget --> 1830000000
# CB test budget --> 

# COMMAND ----------

# DBTITLE 1,Update Every Month
decomp_database_name = "tpo_output_v32"
print(decomp_database_name)
opt_database_name = "ui_output_v32"
print(opt_database_name)
spend_list = [input_params['optimizationInput']['spendtype']]
print("Spend Type = ",spend_list)
business_type = input_params['optimizationInput']['business']
print("Business Type = ",business_type)
channel_list = input_params['optimizationInput']['channel']
print("Channel List = ",channel_list)
category_list = input_params['optimizationInput']['category']
print("Category List = ",category_list)
shisha_list = input_params['optimizationInput']['shisha']
print("Shisha List = ",shisha_list)
scenario_name = input_params['scenarioPrameters']['scenarioName']
print("Scenario Name = ",scenario_name)
obj_name = input_params['optimizationInput']['optimizationobjective']
if obj_name == 'Maximize revenue keeping spends same as last quarter':
    strategy_no = '2'
    sub_strategy_no = None
    goal_spend = None
elif obj_name == 'Achieve maximum growth basis budgeted spend':
    strategy_no = '3'
    sub_strategy_no = '1'
    goal_spend = float(input_params['optimizationInput']['budget'])
year_ = str(int(input_params['optimizationInput']['max'].split('-')[0])-1)
max_month = int(input_params['optimizationInput']['max'].split('-')[1])
min_month = int(input_params['optimizationInput']['min'].split('-')[1])

last_year_period = []
for i in range(max_month,(min_month-1),-1):
    if len(str(i)) == 1:
        val = year_ + '-0' + str(i) + '-01'
    elif len(str(i)) == 2:
        val = year_ + '-' + str(i) + '-01'
    last_year_period.append(val)
print("Last Year Period = ",last_year_period)

# COMMAND ----------

data_prep_iter1 = 'data_prep_iter1_'+scenario_name.replace(" ","")
print(data_prep_iter1)
data_prep_iter2 = 'data_prep_iter2_v2_'+scenario_name.replace(" ","")
print(data_prep_iter2)
decomp_table_name = 'optimization_input_pc'
print(decomp_table_name)
first_iter_data = 'opt_stage1_output_'+strategy_no+'_'+scenario_name.replace(" ","")
print(first_iter_data)
second_iter_data = 'opt_stage2_output_'+strategy_no+'_'+scenario_name.replace(" ","")
print(second_iter_data)

# COMMAND ----------




# COMMAND ----------

try:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.01]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.01]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()

# %md
### Stage 1 Data Prep

# COMMAND ----------

try:
    
    def col_name(col_list, key):
        dict_ = {}
        key_split = key.split("$")
        for i in range(len(col_list)):
            dict_[col_list[i]] = key_split[i]
        return dict_

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.02]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.02]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:

    model_level_remove_list = ['','SHISHA','PLANNINGCUSTOMER']
    coef_less_than_zero = [[(0.975, 0.025), (0.025, 0.975)], [(0.95, 0.05), (0.05, 0.95)], [(0.9, 0.1), (0.1, 0.9)], [(0.85, 0.15), (0.15, 0.85)], [(0.8, 0.2), (0.2, 0.8)], [(0.75, 0.25), (0.25, 0.75)], [(0.7, 0.3), (0.3, 0.7)]]
    coef_more_than_zero = [[(0.975, 0.975), (0.025, 0.025)], [(0.95, 0.95), (0.05, 0.05)], [(0.9, 0.9), (0.1, 0.1)], [(0.85, 0.85), (0.15, 0.15)], [(0.8, 0.8), (0.2, 0.2)], [(0.75, 0.75), (0.25, 0.25)], [(0.7, 0.7), (0.3, 0.3)]]

    for model_level_remove in model_level_remove_list:
        warnings.filterwarnings('ignore') 
        col_list = ['SHISHA','PLANNINGCUSTOMER']
        if model_level_remove == "":
            run_type = "First"
        else:
            run_type = ""
        col_list = [x for x in col_list if x not in model_level_remove]
        print(col_list)
     
        spends_rev = (spark.sql("select *, concat_ws('$', {1}) as Key_Combination from {0}.{2} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD')".format(opt_database_name,str(col_list).replace("[", "").replace("]", "").replace("'", ""),decomp_table_name))
                      .drop('BUSINESS','CATEGORY','SUBCATEGORY','CLUSTER', 'CHANNEL')
                      .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)))
                      .withColumn("STARTDATE",sf.to_date(sf.concat_ws("/",sf.lit("1"),sf.col("Month"),sf.col("Year")),"dd/MM/yyyy"))
                  )
        spends_rev = spends_rev[(spends_rev["SPENDS"] > 0) & (spends_rev["VOLUME"] > 0)].toPandas()
        spends_rev["STARTDATE"] = spends_rev["STARTDATE"].apply(lambda x: str(x))
        
        model_df = pd.DataFrame(columns = ["Key_Combination", *col_list, "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept","Spends_Coef"])
        
        for key in list(spends_rev.Key_Combination.unique()):
    #         print(key)
            temp = spends_rev[(spends_rev["Key_Combination"] == key)].reset_index(drop=True)
            temp['Percentile_Rank_Spends'] = temp.SPENDS.rank(pct = True)
            temp['Percentile_Rank_Volume'] = temp.VOLUME.rank(pct = True)
            x_log = np.log(temp['SPENDS'].fillna(0).values.reshape(-1,1))
            y_log = np.log(temp['VOLUME'].fillna(0).values.reshape(-1,1))
            model = LinearRegression()
            model.fit(x_log, y_log)
            coefficient = model.coef_[0][0]
            org_n_row = temp.shape[0]
            n_row = temp.shape[0]
            r_sq = model.score(x_log, y_log)
            intercept_ = model.intercept_[0]
            intercept_ = max(intercept_, 0.0)
            model_type = "Original"
            if coefficient < 0:
                for comb in coef_less_than_zero:
                    less_temp = temp[~(((temp['Percentile_Rank_Spends']>=comb[0][0]) & (temp['Percentile_Rank_Volume']<=comb[0][1]))|((temp['Percentile_Rank_Spends']<=comb[1][0]) & (temp['Percentile_Rank_Volume']>=comb[1][1])))]
                    x_log_less = np.log(less_temp['SPENDS'].fillna(0).values.reshape(-1,1))
                    y_log_less = np.log(less_temp['VOLUME'].fillna(0).values.reshape(-1,1))
                    if (len(x_log_less) & len(y_log_less))>0:
                        model_less = LinearRegression()
                        model_less.fit(x_log_less, y_log_less)
                        if  0.1 < model_less.coef_[0][0] < 0.9:
                            coefficient = model_less.coef_[0][0]
                            r_sq = model_less.score(x_log_less, y_log_less)
                            intercept_ = model_less.intercept_[0]
                            intercept_ = max(intercept_, 0.0)
                            n_row = less_temp.shape[0]
                            model_type = "augmented_less"+str(comb)
                    else:
                        coefficient = model.coef_[0][0]
                        r_sq = model.score(x_log, y_log)
                        intercept_ = model.intercept_[0]
                        intercept_ = max(intercept_, 0.0)
                        n_row = temp.shape[0]
                        model_type = "Original"

            elif coefficient > 1:
                for comb_gt in coef_more_than_zero:
                    gt_temp = temp[~(((temp['Percentile_Rank_Spends']>=comb_gt[0][0]) & (temp['Percentile_Rank_Volume']>=comb_gt[0][1]))|((temp['Percentile_Rank_Spends']<=comb_gt[1][0]) & (temp['Percentile_Rank_Volume']<=comb_gt[1][1])))]
                    x_log_more = np.log(gt_temp['SPENDS'].fillna(0).values.reshape(-1,1))
                    y_log_more = np.log(gt_temp['VOLUME'].fillna(0).values.reshape(-1,1))
                    if (len(x_log_more) & len(y_log_more))>0:
                        model_more = LinearRegression()
                        model_more.fit(x_log_more, y_log_more)
                        if  0.1 < model_more.coef_[0][0] < 0.9:
                            coefficient = model_more.coef_[0][0]
                            r_sq = model_more.score(x_log_more, y_log_more)
                            intercept_ = model_more.intercept_[0]
                            intercept_ = max(intercept_, 0.0)
                            n_row = gt_temp.shape[0]
                            model_type = "augmented_more"+str(comb_gt)
                    else:
                        coefficient = model.coef_[0][0]
                        r_sq = model.score(x_log, y_log)
                        intercept_ = model.intercept_[0]
                        intercept_ = max(intercept_, 0.0)
                        n_row = temp.shape[0]
                        model_type = "Original"
            else:
                coefficient = model.coef_[0][0]
                r_sq = model.score(x_log, y_log)
                intercept_ = model.intercept_[0]
                intercept_ = max(intercept_, 0.0)
                n_row = temp.shape[0]
                model_type = "Original"

            model_df = model_df.append({"Key_Combination": key, **col_name(col_list, key), "MODELINGNAME": spend_list, "First_Sales_Month": temp['STARTDATE'].min(), "Last_Sales_Month": temp['STARTDATE'].max(), "Number_of_Rows": org_n_row, "Number_of_Rows_after_Perct_Rank": n_row, "Total_Overall_Spends": temp['SPENDS'].sum(), "Total_Overall_Volume": temp['VOLUME'].sum(), "ModelType": model_type, "R_Squared": r_sq, "Intercept": intercept_, "Spends_Coef": coefficient}, ignore_index = True)

        if run_type == 'First':
            final_model_df = model_df.copy(True)
        else:
            if len(col_list) == 2:
                final_model_df['Key_Combination_'+model_level_remove]=final_model_df[col_list[0]]+"$"+final_model_df[col_list[1]]
            elif len(col_list) == 1:
                final_model_df['Key_Combination_'+model_level_remove] = final_model_df[col_list[0]]

            final_model_df = final_model_df.merge(model_df.rename({"Spends_Coef":"Spends_"+model_level_remove+"_Coef", "Key_Combination": 'Key_Combination_'+model_level_remove}, axis=1)[['Key_Combination_'+model_level_remove, "Spends_"+model_level_remove+"_Coef"]], on = 'Key_Combination_'+model_level_remove, how = "left")
            final_model_df["ModelType"] = np.where(~final_model_df["Spends_Coef"].between(0.1, 0.9), "Without_"+model_level_remove+"_Level_Model_Augmented", final_model_df["ModelType"])
            final_model_df["Spends_Coef"] = np.where(~final_model_df["Spends_Coef"].between(0.1, 0.9),
                                                         final_model_df["Spends_"+model_level_remove+"_Coef"],
                                                         final_model_df["Spends_Coef"])
            final_model_df.drop(['Key_Combination_'+model_level_remove, "Spends_"+model_level_remove+"_Coef"], axis = 1, inplace = True)
        #Printing final model shapes
        print(model_df.shape)
        print(final_model_df.shape)
        print(final_model_df[~(final_model_df["Spends_Coef"].between(0.1, 0.9))].shape)
        print("--------------------------------------------------------------------------------------------------------------")
    # final_model_df.display()
    
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.04]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.04]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------

try:

    col_list = ['SHISHA','PLANNINGCUSTOMER']
    print(col_list)

    spends_rev1 = (spark.sql("select *, concat_ws('$', {1}) as Key_Combination from {0}.{2} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD')".format(opt_database_name,str(col_list).replace("[", "").replace("]", "").replace("'", ""),decomp_table_name))
                   .drop('BUSINESS','CATEGORY','SUBCATEGORY','CLUSTER', 'CHANNEL')
                   .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)))
                   .withColumn("STARTDATE",sf.to_date(sf.concat_ws("/",sf.lit("1"),sf.col("Month"),sf.col("Year")),"dd/MM/yyyy"))
                   )
    spends_rev1 = spends_rev1[(spends_rev1["SPENDS"] > 0) & (spends_rev1["VOLUME"] == 0)].toPandas()
    spends_rev1["STARTDATE"] = spends_rev1["STARTDATE"].apply(lambda x: str(x))
    # print(spends_rev1.shape)

    if spends_rev1.shape[0] != 0:
        spend_more_than_0_rev_0 = pd.DataFrame(columns = ["Key_Combination", *col_list, "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept","Spends_Coef"])
            
        for key in list(spends_rev1.Key_Combination.unique()):
        #     print(key)
            temp = spends_rev1[(spends_rev1["Key_Combination"] == key)].reset_index(drop=True)
            model_type = "Spends_more_than_0_and_Volume_0"
            spend_more_than_0_rev_0 = spend_more_than_0_rev_0.append({"Key_Combination": key, **col_name(col_list, key), "MODELINGNAME": spend_list, "First_Sales_Month": "", "Last_Sales_Month": "", "Number_of_Rows": temp.shape[0],"Number_of_Rows_after_Perct_Rank": 0, "Total_Overall_Spends": temp['SPENDS'].sum(), "Total_Overall_Volume": temp['VOLUME'].sum(), "ModelType": model_type, "R_Squared": 0, "Intercept": 0, "Spends_Coef": 0}, ignore_index = True)

        spend_more_than_0_rev_0["R_Squared"] = spend_more_than_0_rev_0["R_Squared"].astype('float')
        spend_more_than_0_rev_0["Intercept"] = spend_more_than_0_rev_0["Intercept"].astype('float')
        spend_more_than_0_rev_0["Spends_Coef"] = spend_more_than_0_rev_0["Spends_Coef"].astype('float')
        # spend_more_than_0_rev_0.display()
        print(spend_more_than_0_rev_0.shape)
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.06]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.06]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    (spark.sql("select * from {0}.{1} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD')".format(opt_database_name,decomp_table_name))
     .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)))
     .filter((sf.col('SPENDS') > 0) & (sf.col('VOLUME') >= 0))
     .select(sf.sum('SPENDS'),sf.sum('VOLUME'))
     ).show()
    if spends_rev1.shape[0] != 0:
        print(final_model_df.Total_Overall_Spends.sum()+spend_more_than_0_rev_0.Total_Overall_Spends.sum())
        print(final_model_df.Total_Overall_Volume.sum()+spend_more_than_0_rev_0.Total_Overall_Volume.sum())
    else:
        print(final_model_df.Total_Overall_Spends.sum())
        print(final_model_df.Total_Overall_Volume.sum())
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.08]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.08]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    if spends_rev1.shape[0] != 0:
        final_model = pd.concat([final_model_df,spend_more_than_0_rev_0], ignore_index = True)
        print(final_model.shape)
        final_model = (final_model
                    .groupby(["Key_Combination","SHISHA","PLANNINGCUSTOMER"])
                    .agg({"First_Sales_Month":sum, "Last_Sales_Month":sum, "Number_of_Rows": sum,
                          "Number_of_Rows_after_Perct_Rank": sum, "Total_Overall_Spends": sum, "Total_Overall_Volume": sum, "ModelType": list, "R_Squared": sum, "Intercept": sum, "Spends_Coef": sum})
                        .reset_index()
                    )
    else:
        final_model = final_model_df.copy(True)
    final_model['MODELINGNAME'] = [spend_list]*final_model.shape[0]
    final_model = final_model[["Key_Combination", "SHISHA", "PLANNINGCUSTOMER", "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept", "Spends_Coef"]]
    print(final_model.Total_Overall_Spends.sum())
    print(final_model.Total_Overall_Volume.sum())
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.10]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.10]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# final_model.display()

# COMMAND ----------


try:
    
    table_name = opt_database_name+'.'+data_prep_iter1
    print(table_name)
    spark.createDataFrame(final_model).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

    print("----------------STAGE1 DATA PREPERATION COMPLETED-------------------------")
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.12]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.12]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# %md
### Stage 2 Data Prep

# COMMAND ----------


try:
    
    
    customer_list = list((spark.sql("select * from {0}.{1}".format(opt_database_name,data_prep_iter1)).select('PLANNINGCUSTOMER').distinct()).toPandas().iloc[:,0])
    customer_list.sort()
    print(len(customer_list))

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.14]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.14]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()


# COMMAND ----------


try:
    
    granularity_df = (spark.sql("select * from {0}.{1}".format(opt_database_name,decomp_table_name))
                      .select('BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
                      .distinct()
                      )
    model_level_remove_list = ['','SHISHA','SEGMENT','PLANNINGCUSTOMER']
    coef_less_than_zero = [[(0.975, 0.025), (0.025, 0.975)], [(0.95, 0.05), (0.05, 0.95)], [(0.9, 0.1), (0.1, 0.9)], [(0.85, 0.15), (0.15, 0.85)], [(0.8, 0.2), (0.2, 0.8)], [(0.75, 0.25), (0.25, 0.75)], [(0.7, 0.3), (0.3, 0.7)]]
    coef_more_than_zero = [[(0.975, 0.975), (0.025, 0.025)], [(0.95, 0.95), (0.05, 0.05)], [(0.9, 0.9), (0.1, 0.1)], [(0.85, 0.85), (0.15, 0.15)], [(0.8, 0.8), (0.2, 0.2)], [(0.75, 0.75), (0.25, 0.25)], [(0.7, 0.7), (0.3, 0.3)]]
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.16]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.16]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------



try:
    
    for model_level_remove in model_level_remove_list:
        warnings.filterwarnings('ignore') 
        col_list = ['SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER']
        if model_level_remove == "":
            run_type = "First"
        else:
            run_type = ""
        col_list = [x for x in col_list if x not in model_level_remove]
        print(col_list)
        # Reading Data
        spends_rev = (spark.sql("select *, concat_ws('$', {1}) as Key_Combination from {0}.{2} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD','LIQUIDATION')".format(opt_database_name,str(col_list).replace("[", "").replace("]", "").replace("'", ""),decomp_table_name))
                  .drop('BUSINESS','CATEGORY','SUBCATEGORY','CLUSTER', 'CHANNEL')
                  .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)) & (sf.col('PLANNINGCUSTOMER').isin(customer_list)))
                  .withColumn("STARTDATE",sf.to_date(sf.concat_ws("/",sf.lit("1"),sf.col("Month"),sf.col("Year")),"dd/MM/yyyy"))
                  )
        spends_rev = spends_rev[(spends_rev["SPENDS"] > 0) & (spends_rev["VOLUME"] > 0)].toPandas()
        spends_rev["STARTDATE"] = spends_rev["STARTDATE"].apply(lambda x: str(x))
        
        model_df = pd.DataFrame(columns = ["Key_Combination", *col_list, "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept","Spends_Coef"])
        
        for key in list(spends_rev.Key_Combination.unique()):
            # print(key)
            temp = spends_rev[(spends_rev["Key_Combination"] == key)].reset_index(drop=True)
            temp['Percentile_Rank_Spends'] = temp.SPENDS.rank(pct = True)
            temp['Percentile_Rank_Volume'] = temp.VOLUME.rank(pct = True)
            x_log = np.log(temp['SPENDS'].fillna(0).values.reshape(-1,1))
            y_log = np.log(temp['VOLUME'].fillna(0).values.reshape(-1,1))
            model = LinearRegression()
            model.fit(x_log, y_log)
            coefficient = model.coef_[0][0]
            org_n_row = temp.shape[0]
            n_row = temp.shape[0]
            r_sq = model.score(x_log, y_log)
            intercept_ = model.intercept_[0]
            intercept_ = max(intercept_, 0.0)
            model_type = "Original"
            if coefficient < 0:
                for comb in coef_less_than_zero:
                    less_temp = temp[~(((temp['Percentile_Rank_Spends']>=comb[0][0]) & (temp['Percentile_Rank_Volume']<=comb[0][1]))|((temp['Percentile_Rank_Spends']<=comb[1][0]) & (temp['Percentile_Rank_Volume']>=comb[1][1])))]
                    x_log_less = np.log(less_temp['SPENDS'].fillna(0).values.reshape(-1,1))
                    y_log_less = np.log(less_temp['VOLUME'].fillna(0).values.reshape(-1,1))
                    if (len(x_log_less) & len(y_log_less))>0:
                        model_less = LinearRegression()
                        model_less.fit(x_log_less, y_log_less)
                        if  0.1 < model_less.coef_[0][0] < 0.9:
                            coefficient = model_less.coef_[0][0]
                            r_sq = model_less.score(x_log_less, y_log_less)
                            intercept_ = model_less.intercept_[0]
                            intercept_ = max(intercept_, 0.0)
                            n_row = less_temp.shape[0]
                            model_type = "augmented_less"+str(comb)
                    else:
                        coefficient = model.coef_[0][0]
                        r_sq = model.score(x_log, y_log)
                        intercept_ = model.intercept_[0]
                        intercept_ = max(intercept_, 0.0)
                        n_row = temp.shape[0]
                        model_type = "Original"

            elif coefficient > 1:
                for comb_gt in coef_more_than_zero:
                    gt_temp = temp[~(((temp['Percentile_Rank_Spends']>=comb_gt[0][0]) & (temp['Percentile_Rank_Volume']>=comb_gt[0][1]))|((temp['Percentile_Rank_Spends']<=comb_gt[1][0]) & (temp['Percentile_Rank_Volume']<=comb_gt[1][1])))]
                    x_log_more = np.log(gt_temp['SPENDS'].fillna(0).values.reshape(-1,1))
                    y_log_more = np.log(gt_temp['VOLUME'].fillna(0).values.reshape(-1,1))
                    if (len(x_log_more) & len(y_log_more))>0:
                        model_more = LinearRegression()
                        model_more.fit(x_log_more, y_log_more)
                        if  0.1 < model_more.coef_[0][0] < 0.9:
                            coefficient = model_more.coef_[0][0]
                            r_sq = model_more.score(x_log_more, y_log_more)
                            intercept_ = model_more.intercept_[0]
                            intercept_ = max(intercept_, 0.0)
                            n_row = gt_temp.shape[0]
                            model_type = "augmented_more"+str(comb_gt)
                    else:
                        coefficient = model.coef_[0][0]
                        r_sq = model.score(x_log, y_log)
                        intercept_ = model.intercept_[0]
                        intercept_ = max(intercept_, 0.0)
                        n_row = temp.shape[0]
                        model_type = "Original"
            else:
                coefficient = model.coef_[0][0]
                r_sq = model.score(x_log, y_log)
                intercept_ = model.intercept_[0]
                intercept_ = max(intercept_, 0.0)
                n_row = temp.shape[0]
                model_type = "Original"

            model_df = model_df.append({"Key_Combination": key, **col_name(col_list, key), "MODELINGNAME": spend_list, "First_Sales_Month": temp['STARTDATE'].min(), "Last_Sales_Month": temp['STARTDATE'].max(), "Number_of_Rows": org_n_row, "Number_of_Rows_after_Perct_Rank": n_row, "Total_Overall_Spends": temp['SPENDS'].sum(), "Total_Overall_Volume": temp['VOLUME'].sum(), "ModelType": model_type, "R_Squared": r_sq, "Intercept": intercept_, "Spends_Coef": coefficient}, ignore_index = True)

        if run_type == 'First':
            final_model_df = model_df.copy(True)
        else:
            if len(col_list) == 2:
                final_model_df['Key_Combination_'+model_level_remove] = final_model_df[col_list[0]]+"$"+final_model_df[col_list[1]]
            elif len(col_list) == 1:
                final_model_df['Key_Combination_'+model_level_remove] = final_model_df[col_list[0]]

            final_model_df = final_model_df.merge(model_df.rename({"Spends_Coef":"Spends_"+model_level_remove+"_Coef", "Key_Combination": 'Key_Combination_'+model_level_remove}, axis=1)[['Key_Combination_'+model_level_remove, "Spends_"+model_level_remove+"_Coef"]], on = 'Key_Combination_'+model_level_remove, how = "left")
            final_model_df["ModelType"] = np.where(~final_model_df["Spends_Coef"].between(0.1, 0.9), "Without_"+model_level_remove+"_Level_Model_Augmented", final_model_df["ModelType"])
            final_model_df["Spends_Coef"] = np.where(~final_model_df["Spends_Coef"].between(0.1, 0.9),
                                                         final_model_df["Spends_"+model_level_remove+"_Coef"],
                                                         final_model_df["Spends_Coef"])
            final_model_df.drop(['Key_Combination_'+model_level_remove, "Spends_"+model_level_remove+"_Coef"], axis = 1, inplace = True)
        #Printing final model shapes
        print(model_df.shape)
        print(final_model_df.shape)
        print(final_model_df[~(final_model_df["Spends_Coef"].between(0.1, 0.9))].shape)
        print("--------------------------------------------------------------------------------------------------------------")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.18]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.18]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# final_model_df.display()

# COMMAND ----------


try:
    
    col_list = ['SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER']
    print(col_list)
    spends_rev1 = (spark.sql("select *, concat_ws('$', {1}) as Key_Combination from {0}.{2} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD','LIQUIDATION')".format(opt_database_name,str(col_list).replace("[", "").replace("]", "").replace("'", ""),decomp_table_name))
                   .drop('BUSINESS','CATEGORY','SUBCATEGORY','CLUSTER', 'CHANNEL')
                   .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)) & (sf.col('PLANNINGCUSTOMER').isin(customer_list)))
                   .withColumn("STARTDATE",sf.to_date(sf.concat_ws("/",sf.lit("1"),sf.col("Month"),sf.col("Year")),"dd/MM/yyyy"))
                   )
    spends_rev1 = spends_rev1[(spends_rev1["SPENDS"] > 0) & (spends_rev1["VOLUME"] == 0)].toPandas()
    spends_rev1["STARTDATE"] = spends_rev1["STARTDATE"].apply(lambda x: str(x))
    # print(spends_rev1.shape)

    if spends_rev1.shape[0] != 0:
        spend_more_than_0_rev_0 = pd.DataFrame(columns = ["Key_Combination", *col_list, "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept","Spends_Coef"])
            
        for key in list(spends_rev1.Key_Combination.unique()):
        #     print(key)
            temp = spends_rev1[(spends_rev1["Key_Combination"] == key)].reset_index(drop=True)
            model_type = "Spends_more_than_0_and_Volume_0"
            spend_more_than_0_rev_0 = spend_more_than_0_rev_0.append({"Key_Combination": key, **col_name(col_list, key), "MODELINGNAME": spend_list, "First_Sales_Month": "", "Last_Sales_Month": "", "Number_of_Rows": temp.shape[0], "Number_of_Rows_after_Perct_Rank": 0, "Total_Overall_Spends": temp['SPENDS'].sum(), "Total_Overall_Volume": temp['VOLUME'].sum(), "ModelType": model_type, "R_Squared": 0, "Intercept": 0, "Spends_Coef": 0}, ignore_index = True)

        spend_more_than_0_rev_0["R_Squared"] = spend_more_than_0_rev_0["R_Squared"].astype('float')
        spend_more_than_0_rev_0["Intercept"] = spend_more_than_0_rev_0["Intercept"].astype('float')
        spend_more_than_0_rev_0["Spends_Coef"] = spend_more_than_0_rev_0["Spends_Coef"].astype('float')
        # spend_more_than_0_rev_0.display()
        print(spend_more_than_0_rev_0.shape)

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.20]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.20]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:
    
    (spark.sql("select * from {0}.{1} where MODELINGNAME in ('PRICEOFF','PROMOTIONAD','LIQUIDATION')".format(opt_database_name,decomp_table_name))
     .drop('BUSINESS','CATEGORY','SUBCATEGORY','CLUSTER', 'CHANNEL')
     .filter((sf.col('MODELINGNAME').isin(spend_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('SHISHA').isin(shisha_list)) & (sf.col('PLANNINGCUSTOMER').isin(customer_list)))
     .filter((sf.col('SPENDS') > 0) & (sf.col('VOLUME') >= 0))
     .select(sf.sum('SPENDS'),sf.sum('VOLUME'))
     ).show()
    if spends_rev1.shape[0] != 0:
        print(final_model_df.Total_Overall_Spends.sum()+spend_more_than_0_rev_0.Total_Overall_Spends.sum())
        print(final_model_df.Total_Overall_Volume.sum()+spend_more_than_0_rev_0.Total_Overall_Volume.sum())
    else:
        print(final_model_df.Total_Overall_Spends.sum())
        print(final_model_df.Total_Overall_Volume.sum())

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.22]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.22]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:
    
    if spends_rev1.shape[0] != 0:
        final_model = pd.concat([final_model_df,spend_more_than_0_rev_0], ignore_index = True)
        print(final_model.shape)
        final_model = (final_model
                       .groupby(['Key_Combination','SEGMENT','SHISHA','PLANNINGCUSTOMER'])
                       .agg({"First_Sales_Month":sum, "Last_Sales_Month":sum, "Number_of_Rows": sum,
                             "Number_of_Rows_after_Perct_Rank": sum, "Total_Overall_Spends": sum, "Total_Overall_Volume": sum, "ModelType": list, "R_Squared": sum, "Intercept": sum, "Spends_Coef": sum})
                       .reset_index()
                       )
    else:
        final_model = final_model_df.copy(True)

    final_model['MODELINGNAME'] = [spend_list]*final_model.shape[0]
    final_model = final_model[["Key_Combination", 'SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER', "MODELINGNAME", "First_Sales_Month", "Last_Sales_Month", "Number_of_Rows", "Number_of_Rows_after_Perct_Rank", "Total_Overall_Spends", "Total_Overall_Volume", "ModelType", "R_Squared", "Intercept","Spends_Coef"]]
    print(final_model.Total_Overall_Spends.sum())
    print(final_model.Total_Overall_Volume.sum())
    # final_model.display()
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.24]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.24]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    opt_output = (spark.createDataFrame(final_model)
                  .join(granularity_df,['SEGMENT','SHISHA','PLANNINGCUSTOMER'],'left')
                  .distinct()
                  .select('Key_Combination', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'First_Sales_Month', 'Last_Sales_Month', 'Number_of_Rows', 'Number_of_Rows_after_Perct_Rank', 'Total_Overall_Spends', 'Total_Overall_Volume', 'ModelType', 'R_Squared', 'Intercept', 'Spends_Coef')
                  )
    # opt_output.display()

    table_name = opt_database_name+"."+data_prep_iter2
    print(table_name)
    opt_output.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)
    print("----------------STAGE2 DATA PREPERATION COMPLETED-------------------------")

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.26]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.26]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------





# %md
### Specifying Input Lists

# COMMAND ----------

try:
    
    iter1_input_list = ['All', strategy_no, sub_strategy_no, 'All', 1.1, 0.9, False, False, False, None, None, None, goal_spend]
    print("First Iteration Input List: ", iter1_input_list)
    iter2_input_list = ['All', '2', None, 'All', 1.1, 0.9, False, False, False, None, None, None, None]
    print("Second Iteration Input List: ", iter2_input_list)
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.28]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.28]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------

# DBTITLE 1,ns_func


def Spend_func(z):
    res = z.sum()
    return res

def Vol_func(z):
    res = (-(opti_data['BackCalc_Intercept']*(z**opti_data['Spends_Coef']))).sum()
    return res

def NS_func(z):
    # res = (-((opti_data['BackCalc_Intercept']*(z**opti_data['Spends_Coef']))*opti_data['PRICE']*opti_data['NS_TO_GPS']*opti_data['Adjustment_Factor'])).sum()
    try:
        spend_change = (z - opti_data['Last_Year_Period_Spends_old'])/opti_data['Last_Year_Period_Spends_old']
        optimized_volume = opti_data["Last_Year_Period_Volume_old"]*(1+spend_change*opti_data["Spends_Coef"])
        Optimized_GPS = optimized_volume*opti_data['Last_Year_Period_GPS_old']/opti_data['Last_Year_Period_Volume_old']
        # res = (-(Optimized_GPS-z)).sum()
    except:
        spend_change = (z - opti_data['Last_Year_Period_Spends'])/opti_data['Last_Year_Period_Spends']
        optimized_volume = opti_data["Last_Year_Period_Volume"]*(1+spend_change*opti_data["Spends_Coef"])
        Optimized_GPS = optimized_volume*opti_data['Last_Year_Period_GPS']/opti_data['Last_Year_Period_Volume']

    res = (-(Optimized_GPS-z)).sum()
    # ns_change = ((Optimized_GPS-z) - opti_data['Last_Year_Period_NetSales'])/opti_data['Last_Year_Period_NetSales'].abs()
    # if ((ns_change < 0) & (spend_change >= 0)).any():
    #     res = (-opti_data['Last_Year_Period_NetSales']).sum()
    # else:
    #     res = (-(Optimized_GPS-z)).sum()
    # # opti_data = opti_data.drop(['Optimized_Volume', 'Optimized_GPS'], axis = 1)

    # return res
    return res

# COMMAND ----------

# DBTITLE 1,netsales_logical_constraint
def netsales_logical_constraint(z):

    # Calculate optimized NetSales
    try:
        ly_spends = opti_data['Last_Year_Period_Spends_old'].values
        ly_netsales = opti_data['Last_Year_Period_NetSales_old'].values
        spend_change = (z - ly_spends) / ly_spends
        optimized_volume = opti_data["Last_Year_Period_Volume_old"]*(1+spend_change*opti_data["Spends_Coef"])
        Optimized_GPS = optimized_volume*opti_data['Last_Year_Period_GPS_old']/opti_data['Last_Year_Period_Volume_old']
    except:
        ly_spends = opti_data['Last_Year_Period_Spends'].values
        ly_netsales = opti_data['Last_Year_Period_NetSales'].values
        spend_change = (z - ly_spends) / ly_spends
        optimized_volume = opti_data["Last_Year_Period_Volume"]*(1+spend_change*opti_data["Spends_Coef"])
        Optimized_GPS = optimized_volume*opti_data['Last_Year_Period_GPS']/opti_data['Last_Year_Period_Volume']
    optimized_ns = (Optimized_GPS - z)

    netsales_change = (optimized_ns - ly_netsales) / np.abs(ly_netsales)

    # Logical condition: (NS_change >= 0) OR (NS_change < 0 AND Spend_change < 0)
    valid = (netsales_change >= 0) | ((netsales_change < 0) & (spend_change < 0))

    # We want all values in `valid` to be True → valid.astype(int) - 1 == 0
    return valid.astype(int) - 1


# COMMAND ----------

# DBTITLE 1,optim_details


def optim_details(argslist):
    strategy_name = argslist[0]
    sub_strategy_name = argslist[1]
    cons = argslist[2]
    bound = argslist[3]
    initials = argslist[4]
    activity_type = argslist[5]
    opti_data = argslist[6]
    step = argslist[7][0]
    epsilon = argslist[7][1]

    ## added this 2 lines
    # nl_constraint = NonlinearConstraint(netsales_logical_constraint, 0, 0)
    # cons = cons + (nl_constraint,)

    if epsilon == True:
        minimizer_kwargs = dict(bounds = bound, constraints = cons, options={'eps': 1.4901161193847656e-10})
        # minimizer_kwargs = dict(bounds = bound, constraints = cons, options={'finite_diff_rel_step': 1e-4}, method = 'trust-constr')
    else:
        # minimizer_kwargs = dict(bounds = bound, constraints = cons, method = 'trust-constr')
        minimizer_kwargs = dict(bounds = bound, constraints = cons)
    print(minimizer_kwargs)
    
    if strategy_name == "strategy1":
            res = ot.basinhopping(Spend_func, x0=initials, minimizer_kwargs=minimizer_kwargs, stepsize = step, seed=0)
    elif strategy_name == "strategy2":
            res = ot.basinhopping(NS_func, x0=initials, minimizer_kwargs=minimizer_kwargs, stepsize = step, seed=0)
    else:
        if sub_strategy_name == "sub-strategy1":
            res = ot.basinhopping(NS_func, x0=initials, minimizer_kwargs=minimizer_kwargs, stepsize = step, seed=0)
        elif sub_strategy_name == "sub-strategy2":
            res = ot.basinhopping(Spend_func, x0=initials, minimizer_kwargs=minimizer_kwargs, stepsize = step, seed=0)
        elif sub_strategy_name == "sub-strategy3":
            res = ot.basinhopping(NS_func, x0=initials, minimizer_kwargs=minimizer_kwargs, stepsize = step, seed=0)
    return res.x

# COMMAND ----------

# def step_size_finder(
#     strategy_name,
#     sub_strategy_name,
#     cons,
#     bound,
#     ini_val,
#     activity_type,
#     dat,
#     goal_perc_revenue,
#     goal_spend,
#     iter_no,
# ):
#     strategy_name = strategy_name
#     sub_strategy_name = sub_strategy_name
#     opti_data = dat
#     activity_type = activity_type
#     cons = cons
#     bound = bound
#     initials = ini_val
#     goal_perc_revenue = goal_perc_revenue
#     goal_spend = goal_spend
#     outputs = []
#     # step_size_list = [[0.25, True],[0.25, False],[0.3, True],[0.35, False],[0.4, True],[0.4, False],[0.45, True],[0.45, False], [0.5, True],[0.5, False]]
#     step_size_list = [
#         [0.45, False],
#         [0.45, True],
#         [0.5, False],
#         [0.5, True],
#         [0.55, False],
#         [0.55, True],
#         [0.6, False],
#         [0.6, True],
#         [0.65, False],
#         [0.65, True],
#         [0.7, False],
#         [0.7, True],
#     ]

#     list_ = [
#         (
#             strategy_name,
#             sub_strategy_name,
#             cons,
#             bound,
#             initials,
#             activity_type,
#             dat,
#             step,
#         )
#         for step in step_size_list
#     ]
#     # print("list",list_)
#     with concurrent.futures.ProcessPoolExecutor(
#         max_workers=len(step_size_list)
#     ) as executor:
#         for rs in executor.map(optim_details, list_):
#             # print(rs)
#             outputs.append(list(rs))
#     # print(outputs)

#     result_df = pd.DataFrame({"stepsize": step_size_list, "optimum": outputs})
#     # print(result_df)
#     step_size = []
#     optimum = []
#     grow = []
#     spends = []
#     overall_grow = []
#     overall_spends = []
#     for i in range(result_df.shape[0]):
#         opti_data["Optimized_Spends"] = result_df.iloc[i, 1]

#         # ## added
#         # if iter_no == "2":
#         #     opti_data["Overall_Spends_change"] = (
#         #         opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends_old"]
#         #     ) / opti_data["Last_Year_Period_Spends_old"]
#         #     ## row wise spends should not exceed +/- 20% at segment level
#         #     opti_data.loc[
#         #         (opti_data["Overall_Spends_change"] > 0.2), "Optimized_Spends"
#         #     ] = opti_data["Last_Year_Period_Spends_old"] * (1 + 0.2)
#         #     opti_data.loc[
#         #         (opti_data["Overall_Spends_change"] < -0.2), "Optimized_Spends"
#         #     ] = opti_data["Last_Year_Period_Spends_old"] * (1 - 0.2)
#         #     # print("iter 2 spends adjusted")
#         #     # print("-------------------------------")
#         # else:
#         #     opti_data["Overall_Spends_change"] = (
#         #         opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends"]
#         #     ) / opti_data["Last_Year_Period_Spends"]
#         #     ## row wise spends should not exceed +/- 10% at customer level
#         #     opti_data.loc[
#         #         (opti_data["Overall_Spends_change"] > 0.1), "Optimized_Spends"
#         #     ] = opti_data["Last_Year_Period_Spends"] * (1 + 0.1)
#         #     opti_data.loc[
#         #         (opti_data["Overall_Spends_change"] < -0.1), "Optimized_Spends"
#         #     ] = opti_data["Last_Year_Period_Spends"] * (1 - 0.1)
#         # # result_df.iloc[i,1] = opti_data["Optimized_Spends"].to_list()
#         # ## till here

#         # opti_data['Optimized_NS'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])*opti_data['PRICE']*opti_data['NS_TO_GPS']*opti_data['Adjustment_Factor']
#         if iter_no == "2":
#             opti_data["Spend_Change"] = (
#                 opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends_old"]
#             ) / opti_data["Last_Year_Period_Spends_old"]
#             opti_data["Optimized_Volume"] = opti_data["Last_Year_Period_Volume_old"] * (
#                 1 + opti_data["Spend_Change"] * opti_data["Spends_Coef"]
#             )
#             opti_data["Optimized_GPS"] = (
#                 opti_data["Optimized_Volume"]
#                 * opti_data["Last_Year_Period_GPS_old"]
#                 / opti_data["Last_Year_Period_Volume_old"]
#             )
#         else:
#             opti_data["Spend_Change"] = (
#                 opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends"]
#             ) / opti_data["Last_Year_Period_Spends"]
#             opti_data["Optimized_Volume"] = opti_data["Last_Year_Period_Volume"] * (
#                 1 + opti_data["Spend_Change"] * opti_data["Spends_Coef"]
#             )
#             opti_data["Optimized_GPS"] = (
#                 opti_data["Optimized_Volume"]
#                 * opti_data["Last_Year_Period_GPS"]
#                 / opti_data["Last_Year_Period_Volume"]
#             )
#         opti_data["Optimized_NS"] = (
#             opti_data["Optimized_GPS"] - opti_data["Optimized_Spends"]
#         )

#         # ## added
#         # if iter_no == '2':
#         #     opti_data['Overall_NetSales_change'] = (opti_data['Optimized_NS'] - opti_data['Last_Year_Period_NetSales_old'])/opti_data['Last_Year_Period_NetSales_old']
#         # else:
#         #     opti_data['Overall_NetSales_change'] = (opti_data['Optimized_NS'] - opti_data['Last_Year_Period_NetSales'])/opti_data['Last_Year_Period_NetSales'].abs()
#         # Skip loop for ANY optimized spend if its positive change causes negative netsales change
#         # if ((opti_data['Overall_NetSales_change'] < 0) & (opti_data['Overall_Spends_change'] >= 0)).any():
#         #     print("POSITIVE SPENDS CHANGE YIELDING NEGATIVE NETSALES CHANGE FOR SPENDS ", result_df.iloc[i,1], "AND NETSALES ", opti_data['Optimized_NS'].to_list())
#         #     print("SKIPPING THIS ITERATION")
#         #     print("-------------------------------------------------------------")
#         #     continue  # Skip this i, move to next iteration of the outer loop
#         # print("SPENDS ", result_df.iloc[i,1], "YIELDS NETSALES ", opti_data['Optimized_NS'].to_list())
#         # print("-------------------------------------------------------------")
#         # ## till here

#         # opti_data['Optimized_Volume'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])
#         step_size.append(result_df.iloc[i, 0])
#         optimum.append(result_df.iloc[i, 1])
#         grow.append(
#             (
#                 (
#                     opti_data.Optimized_NS.sum()
#                     - opti_data.Last_Year_Period_NetSales.sum()
#                 )
#                 / opti_data.Last_Year_Period_NetSales.sum()
#             )
#         )
#         # grow.append(((opti_data.Optimized_Volume.sum() - opti_data.Last_Year_Period_Volume.sum())/opti_data.Last_Year_Period_Volume.sum()))
#         spends.append(
#             (
#                 (
#                     opti_data.Optimized_Spends.sum()
#                     - opti_data.Last_Year_Period_Spends.sum()
#                 )
#                 / opti_data.Last_Year_Period_Spends.sum()
#             )
#         )

#         # if iter_no == '2':
#         #     overall_grow.append(((opti_data.Optimized_NS.sum() - opti_data.Last_Year_Period_NetSales_old.sum())/opti_data.Last_Year_Period_NetSales_old.sum()))
#         #     overall_spends.append(((opti_data.Optimized_Spends.sum() - opti_data.Last_Year_Period_Spends_old.sum())/opti_data.Last_Year_Period_Spends_old.sum()))

#         opti_data = opti_data.drop(
#             [
#                 "Optimized_Spends",
#                 "Optimized_NS",
#                 "Optimized_Volume",
#                 "Optimized_GPS",
#                 # "Overall_Spends_change",
#                 "Spend_Change",
#             ],
#             axis=1,
#         )
#         # opti_data.drop(['Optimized_Spends', 'Optimized_Volume'], axis = 1)
#         # print(opti_data.columns)

#     df_all = pd.DataFrame(
#         {
#             "Step": step_size,
#             "Optimum": optimum,
#             "Growth_percent": grow,
#             "Spend_percent": spends,
#         }
#     )
#     # if iter_no == '2':
#     #     df_all = pd.DataFrame({"Step": step_size, "Optimum": optimum, "Growth_percent": grow, "Spend_percent": spends, "Overall_Growth_percent": overall_grow, "Overall_Spend_percent": overall_spends})
#     # else:
#     #     df_all = pd.DataFrame({"Step": step_size, "Optimum": optimum, "Growth_percent": grow, "Spend_percent": spends, "Overall_Growth_percent": grow, "Overall_Spend_percent": spends})

#     if goal_spend == None:
#         df_all = df_all[df_all["Spend_percent"] <= 0.005].reset_index(drop=True)
#         # df_all = df_all[(df_all["Overall_Spend_percent"] >= -0.2) & (df_all["Overall_Spend_percent"] <= 0.2)]
#         print(spark.createDataFrame(df_all).count())
#         # df_all = df_all[(df_all['Overall_Growth_percent'] >= 0) | ((df_all['Overall_Growth_percent'] < 0) & (df_all['Overall_Spend_percent'] < 0))].reset_index(drop=True) ### this is the new line

#     df_all = df_all.sort_values(by="Growth_percent", ascending=False)
#     # df_all.display()
#     print(df_all.iloc[0, 0])
#     # print(df_all.iloc[0, 1])
#     return df_all.iloc[0, 1]

# COMMAND ----------

# DBTITLE 1,step_size_finder
def step_size_finder(
    strategy_name,
    sub_strategy_name,
    cons,
    bound,
    ini_val,
    activity_type,
    dat,
    goal_perc_revenue,
    goal_spend,
    iter_no,
):
    strategy_name = strategy_name
    sub_strategy_name = sub_strategy_name
    opti_data = dat
    activity_type = activity_type
    cons = cons
    bound = bound
    initials = ini_val
    goal_perc_revenue = goal_perc_revenue
    goal_spend = goal_spend
    outputs = []
    # step_size_list = [[0.25, True],[0.25, False],[0.3, True],[0.35, False],[0.4, True],[0.4, False],[0.45, True],[0.45, False], [0.5, True],[0.5, False]]
    step_size_list = [
        [0.45, False],
        [0.45, True],
        [0.5, False],
        [0.5, True],
        [0.55, False],
        [0.55, True],
        [0.6, False],
        [0.6, True],
        [0.65, False],
        [0.65, True],
        [0.7, False],
        [0.7, True],
    ]

    list_ = [
        (
            strategy_name,
            sub_strategy_name,
            cons,
            bound,
            initials,
            activity_type,
            dat,
            step,
        )
        for step in step_size_list
    ]
    # print("list",list_)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(step_size_list)
    ) as executor:
        for rs in executor.map(optim_details, list_):
            # print(rs)
            outputs.append(list(rs))
    # print(outputs)

    result_df = pd.DataFrame({"stepsize": step_size_list, "optimum": outputs})
    # print(result_df)
    step_size = []
    optimum = []
    grow = []
    spends = []
    overall_grow = []
    overall_spends = []
    for i in range(result_df.shape[0]):
        opti_data["Optimized_Spends"] = result_df.iloc[i, 1]

        # opti_data['Optimized_NS'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])*opti_data['PRICE']*opti_data['NS_TO_GPS']*opti_data['Adjustment_Factor']
        if iter_no == "2":
            opti_data["Spend_Change"] = (
                opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends_old"]
            ) / opti_data["Last_Year_Period_Spends_old"]
            opti_data["Optimized_Volume"] = opti_data["Last_Year_Period_Volume_old"] * (
                1 + opti_data["Spend_Change"] * opti_data["Spends_Coef"]
            )
            opti_data["Optimized_GPS"] = (
                opti_data["Optimized_Volume"]
                * opti_data["Last_Year_Period_GPS_old"]
                / opti_data["Last_Year_Period_Volume_old"]
            )
        else:
            opti_data["Spend_Change"] = (
                opti_data["Optimized_Spends"] - opti_data["Last_Year_Period_Spends"]
            ) / opti_data["Last_Year_Period_Spends"]
            opti_data["Optimized_Volume"] = opti_data["Last_Year_Period_Volume"] * (
                1 + opti_data["Spend_Change"] * opti_data["Spends_Coef"]
            )
            opti_data["Optimized_GPS"] = (
                opti_data["Optimized_Volume"]
                * opti_data["Last_Year_Period_GPS"]
                / opti_data["Last_Year_Period_Volume"]
            )
        opti_data["Optimized_NS"] = (
            opti_data["Optimized_GPS"] - opti_data["Optimized_Spends"]
        )

        # opti_data['Optimized_Volume'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])
        step_size.append(result_df.iloc[i, 0])
        optimum.append(result_df.iloc[i, 1])
        # if iter_no == "2":
        #     grow.append(
        #         (
        #             (
        #                 opti_data.Optimized_NS.sum()
        #                 - opti_data.Last_Year_Period_NetSales_old.sum()
        #             )
        #             / opti_data.Last_Year_Period_NetSales_old.sum()
        #         )
        #     )
        # else:
        #     grow.append(
        #         (
        #             (
        #                 opti_data.Optimized_NS.sum()
        #                 - opti_data.Last_Year_Period_NetSales.sum()
        #             )
        #             / opti_data.Last_Year_Period_NetSales.sum()
        #         )
        #     )
        grow.append(
            (
                (
                    opti_data.Optimized_NS.sum()
                    - opti_data.Last_Year_Period_NetSales.sum()
                )
                / opti_data.Last_Year_Period_NetSales.sum()
            )
        )
        # grow.append(((opti_data.Optimized_Volume.sum() - opti_data.Last_Year_Period_Volume.sum())/opti_data.Last_Year_Period_Volume.sum()))
        spends.append(
            (
                (
                    opti_data.Optimized_Spends.sum()
                    - opti_data.Last_Year_Period_Spends.sum()
                )
                / opti_data.Last_Year_Period_Spends.sum()
            )
        )

        if iter_no == '2':
            # overall_grow.append(((opti_data.Optimized_NS.sum() - opti_data.Last_Year_Period_NetSales_old.sum())/opti_data.Last_Year_Period_NetSales_old.sum()))
            overall_spends.append(((opti_data.Optimized_Spends.sum() - opti_data.Last_Year_Period_Spends_old.sum())/opti_data.Last_Year_Period_Spends_old.sum()))
        else:
            overall_spends.append(0)

        opti_data = opti_data.drop(
            [
                "Optimized_Spends",
                "Optimized_NS",
                "Optimized_Volume",
                "Optimized_GPS",
                "Spend_Change",
            ],
            axis=1,
        )
        # opti_data.drop(['Optimized_Spends', 'Optimized_Volume'], axis = 1)
        # print(opti_data.columns)

    df_all = pd.DataFrame(
        {
            "Step": step_size,
            "Optimum": optimum,
            "Growth_percent": grow,
            "Spend_percent": spends,
            "Overall_Spend_percent": overall_spends,
        }
    )

    if goal_spend == None:
        df_all = df_all[df_all["Spend_percent"] <= 0.005].reset_index(drop=True)
        # df_all.display()
        if iter_no== '2':
            df_all = df_all[df_all["Overall_Spend_percent"].abs().round(1) <= 0.1].reset_index(drop=True) # overall cust level bounds
        print(spark.createDataFrame(df_all).count())
        # df_all = df_all[(df_all['Overall_Growth_percent'] >= 0) | ((df_all['Overall_Growth_percent'] < 0) & (df_all['Overall_Spend_percent'] < 0))].reset_index(drop=True) ### this is the new line

    if iter_no == '2':
        df_all = df_all.sort_values(by=["Spend_percent", "Growth_percent"], ascending=[False, False])
    else:
        df_all = df_all.sort_values(by="Growth_percent", ascending=False)
    # df_all.display()
    print(df_all.iloc[0, 0])
    # print(df_all.iloc[0, 1])
    return df_all.iloc[0, 1]

# COMMAND ----------

# DBTITLE 1,optimizer_master



def optimizer_master(List, dat, iter_no):
    optimization_category = List[0]
    strategy_name = "strategy" + List[1]
    sub_strategy = List[2]
    
    if  sub_strategy != None:
        sub_strategy_name = "sub-strategy" + List[2]
    
    activity_type = List[3]
    ub_overall = List[4]
    lb_overall = List[5]
    custom_bounds = List[6]
    new_constraints = List[7]
    goal_based = List[8]
    bounds_dict = List[9]
    cons_df = List[10]
    goal_perc_revenue = List[11]
    goal_spend = List[12]
    opti_data = dat
    iter_no = iter_no
    print(opti_data.shape)
    
#     specifying hyperparameters for optimization

    initials = np.array(opti_data['Last_Year_Period_Spends'])
    custom_lb = [lb_overall]*len(initials)
    custom_ub = [ub_overall]*len(initials)
    bound = [((initials[i]*custom_lb[i]), (initials[i]*custom_ub[i])) for i in range(len(initials))]
    
#     Optimization for different strategies
    
    if strategy_name == "strategy1":
        sub_strategy_name = 'Not Available'
        c1 = NonlinearConstraint(NS_func, lb=(-np.inf), ub=(-opti_data['Last_Year_Period_NetSales'].sum()))
        l_S1 = [c1]
        cons = tuple(l_S1)
        print(cons)
    elif strategy_name == "strategy2":
        sub_strategy_name = 'Not Available'
        c2 = LinearConstraint(np.ones(len(initials)), lb=0, ub=opti_data['Last_Year_Period_Spends'].sum())
        # c2 = LinearConstraint(np.ones(len(initials)), lb=(opti_data['Last_Year_Period_Spends'].sum())*(1-0.1), ub=(opti_data['Last_Year_Period_Spends'].sum())*(1+0.1))
        l_S2 = [c2]
        if iter_no == '2':
            c21 = LinearConstraint(np.ones(len(initials)), lb=(opti_data['Last_Year_Period_Spends_old'].sum())*(1-0.1), ub=(opti_data['Last_Year_Period_Spends_old'].sum())*(1+0.1)) ## changed from 0.2
            l_S2.append(c21)
        cons = tuple(l_S2)
        print(cons)
    elif strategy_name == "strategy3":
        if sub_strategy_name == "sub-strategy1":
            c31 = LinearConstraint(np.ones(len(initials)), lb = goal_spend, ub = goal_spend) 
            l_S31 = [c31]
            if iter_no == '2':
                c311 = LinearConstraint(np.ones(len(initials)), lb=(opti_data['Last_Year_Period_Spends_old'].sum())*(1-0.2), ub=(opti_data['Last_Year_Period_Spends_old'].sum())*(1+0.2))
                l_S31.append(c311)
            cons = tuple(l_S31)
            print(cons)
        elif sub_strategy_name == "sub-strategy2":
            c32 = NonlinearConstraint(NS_func, lb = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)), ub = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)))
            # c32_1 = LinearConstraint(np.ones(len(initials)), lb = (opti_data['Last_Year_Period_Spends'].sum()*(1-0.2)), ub = (opti_data['Last_Year_Period_Spends'].sum()))
            l_S32 = [c32]
            # l_S32.append(c32_1)
            cons = tuple(l_S32)
            print(cons)
        elif sub_strategy_name == "sub-strategy3":
            c33_1 = NonlinearConstraint(Vol_func, lb = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)), ub = (-opti_data['Last_Year_Period_NetSales'].sum()))
            c33_2 = LinearConstraint(np.ones(len(initials)), lb = goal_spend, ub = goal_spend)
            l_S33 = [c33_1]
            l_S33.append(c33_2)
            cons = tuple(l_S33)
            print(cons)
    res = step_size_finder(strategy_name, sub_strategy_name, cons, bound, initials, activity_type, opti_data, goal_perc_revenue, goal_spend, iter_no)     
    return custom_lb, custom_ub, res

# COMMAND ----------

# decomp_df_0_0 = decomp_df_base.withColumn(
#     "STARTDATE",
#     sf.to_date(
#         sf.concat_ws("/", sf.lit("01"), sf.col("MONTH"), sf.col("YEAR")), "dd/MM/yyyy"
#     ),
# ).filter(
#     (sf.col("STARTDATE").isin(last_year_period))
#     & (sf.col("MODELINGNAME").isin(spend_list))
#     & (sf.col("BUSINESS") == business_type)
#     & (sf.col("CATEGORY").isin(category_list))
#     & (sf.col("CHANNEL").isin(channel_list))
#     & (sf.col("SHISHA").isin(shisha_list))
#     & (sf.col("SPENDS") > 0)
#     & (sf.col("VOLUME") >= 0)


# %md
### Optimization at Shisha x Customer Level

# COMMAND ----------

spark.sql("select * from {0}.{1}".format(opt_database_name,decomp_table_name)).withColumn(
    "STARTDATE",
    sf.to_date(
        sf.concat_ws("/", sf.lit("01"), sf.col("MONTH"), sf.col("YEAR")), "dd/MM/yyyy"
    ),
).filter(
    (sf.col("STARTDATE").isin(last_year_period))
    & (sf.col("MODELINGNAME").isin(spend_list))
    & (sf.col("BUSINESS") == business_type)
    & (sf.col("CATEGORY").isin(category_list))
    & (sf.col("CHANNEL").isin(channel_list))
    & (sf.col("SHISHA").isin(shisha_list))
    & (sf.col("SPENDS") > 0)
    & (sf.col("VOLUME") > 0)).filter("PLANNINGCUSTOMER == 'L5 JP NS Marui Honbu' and SEGMENT == 'JP/C&B/RTD/LargePET/PPP PET'").agg(sf.sum("NETSALES")).display()

# COMMAND ----------

decomp_table_name

# COMMAND ----------

# DBTITLE 1,iter 1
try:
    
    decomp_df = spark.sql("select * from {0}.{1}".format(opt_database_name,"optimization_input"))
    coef_info_df = spark.sql("select * from {0}.{1}".format(opt_database_name,data_prep_iter1))
    # category_list = list(coef_info_df.select('CATEGORY').distinct().toPandas().iloc[0,0])
    # print(category_list)
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.30]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.30]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------

try:

    decomp_df = (
        decomp_df.withColumn(
            "STARTDATE",
            sf.to_date(
                sf.concat_ws("/", sf.lit("01"), sf.col("Month"), sf.col("Year")),
                "dd/MM/yyyy",
            ),
        )
        .filter(
            (sf.col("STARTDATE").isin(last_year_period))
            & (sf.col("MODELINGNAME").isin(spend_list))
            & (sf.col("BUSINESS") == business_type)
            & (sf.col("CATEGORY").isin(category_list))
            & (sf.col("CHANNEL").isin(channel_list))
            & (sf.col("SHISHA").isin(shisha_list))
            & (sf.col("SPENDS") > 0)
            & (sf.col("VOLUME") >= 0)
        )
        .groupBy("SHISHA", "CHANNEL", "PLANNINGCUSTOMER")
        .agg(
            sf.sum("SPENDS").alias("SPENDS"),
            sf.sum("VOLUME").alias("VOLUME"),
            sf.sum("GPS").alias("GPS"),
            sf.sum("NETSALES").alias("NETSALES"),
            sf.mean("PRICE").alias("PRICE"),
        )
        .withColumn("SPENDS", sf.round(sf.col("SPENDS"), 3))
        .withColumn("VOLUME", sf.round(sf.col("VOLUME"), 3))
        .withColumn("GPS", sf.round(sf.col("GPS"), 3))
        .withColumn("NETSALES", sf.round(sf.col("NETSALES"), 3))
        .withColumn("PRICE", sf.round(sf.col("PRICE"), 3))
    )
    # decomp_df.display()
    print("decomp_df created")

    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
                "PROGRESSMESSAGE": [0.32],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()
except:
    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [2],
                "PROGRESSMESSAGE": [0.32],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()

# COMMAND ----------


try:
    
    data_to_opt = (coef_info_df
                   .join(decomp_df,['SHISHA','PLANNINGCUSTOMER'], 'left')
                   .fillna(0, subset = ['SPENDS','VOLUME','GPS','NETSALES','PRICE'])
                   .withColumnRenamed('SPENDS','Last_Year_Period_Spends')
                   .withColumnRenamed('VOLUME','Last_Year_Period_Volume')
                   .withColumnRenamed('GPS','Last_Year_Period_GPS')
                   .withColumnRenamed('NETSALES','Last_Year_Period_NetSales')
                   .withColumn('Last_Year_Period_NetSales', sf.col('Last_Year_Period_GPS')-sf.col('Last_Year_Period_Spends'))
                #    .withColumn('MC_PER_NETSALES',sf.when(sf.col('Last_Year_Period_NetSales')!=0,sf.round(sf.col('MC')/sf.col('Last_Year_Period_NetSales'),3)).otherwise(sf.lit(0).cast('double')))
                   .withColumn('NS_TO_GPS', sf.when(sf.col('Last_Year_Period_GPS')!=0,sf.round(sf.col('Last_Year_Period_NetSales')/sf.col('Last_Year_Period_GPS'),3)).otherwise(sf.lit(0)))
                   )
    # data_to_opt.display()
    print("data_to_opt created")

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.34]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.34]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:
    
    
    opti_data = (data_to_opt
                 .filter((sf.col('Last_Year_Period_Spends')>0) & (sf.col('Last_Year_Period_Volume')>0))
                 .select('SHISHA', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'PRICE', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS', 'Intercept')
                 .toPandas()
                 )
    non_opt_data = (data_to_opt
                    .filter((sf.col('Last_Year_Period_Spends')>0) & (sf.col('Last_Year_Period_Volume')==0))
                    .select('SHISHA', 'CHANNEL', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'PRICE', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS')
                    .toPandas()
                    )
    spark.createDataFrame(opti_data).display()
    print("opti_data created")
    if non_opt_data.shape[0] != 0:
        non_opt_data.display()
        print("non_opt_data created")

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.36]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.36]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------

# List = iter1_input_list
# dat = opti_data
# iter_no = 1

# optimization_category = List[0]
# strategy_name = "strategy" + List[1]
# sub_strategy = List[2]

# if  sub_strategy != None:
#     sub_strategy_name = "sub-strategy" + List[2]

# activity_type = List[3]
# ub_overall = List[4]
# lb_overall = List[5]
# custom_bounds = List[6]
# new_constraints = List[7]
# goal_based = List[8]
# bounds_dict = List[9]
# cons_df = List[10]
# goal_perc_revenue = List[11]
# goal_spend = List[12]
# opti_data = dat
# iter_no = iter_no
# print(opti_data.shape)

# #     specifying hyperparameters for optimization

# initials = np.array(opti_data['Last_Year_Period_Spends'])
# custom_lb = [lb_overall]*len(initials)
# custom_ub = [ub_overall]*len(initials)
# bound = [((initials[i]*custom_lb[i]), (initials[i]*custom_ub[i])) for i in range(len(initials))]

# #     Optimization for different strategies

# if strategy_name == "strategy1":
#     sub_strategy_name = 'Not Available'
#     c1 = NonlinearConstraint(NS_func, lb=(-np.inf), ub=(-opti_data['Last_Year_Period_NetSales'].sum()))
#     l_S1 = [c1]
#     cons = tuple(l_S1)
#     print(cons)
# elif strategy_name == "strategy2":
#     sub_strategy_name = 'Not Available'
#     c2 = LinearConstraint(np.ones(len(initials)), lb=0, ub=opti_data['Last_Year_Period_Spends'].sum())
#     l_S2 = [c2]
#     if iter_no == '2':
#         c21 = LinearConstraint(np.ones(len(initials)), lb=(opti_data['Last_Year_Period_Spends_old'].sum())*(1-0.2), ub=(opti_data['Last_Year_Period_Spends_old'].sum())*(1+0.2)) ## changed from 0.2
#         l_S2.append(c21)
#     cons = tuple(l_S2)
#     print(cons)
# elif strategy_name == "strategy3":
#     if sub_strategy_name == "sub-strategy1":
#         c31 = LinearConstraint(np.ones(len(initials)), lb = goal_spend, ub = goal_spend) 
#         l_S31 = [c31]
#         if iter_no == '2':
#             c311 = LinearConstraint(np.ones(len(initials)), lb=(opti_data['Last_Year_Period_Spends_old'].sum())*(1-0.2), ub=(opti_data['Last_Year_Period_Spends_old'].sum())*(1+0.2))
#             l_S31.append(c311)
#         cons = tuple(l_S31)
#         print(cons)
#     elif sub_strategy_name == "sub-strategy2":
#         c32 = NonlinearConstraint(NS_func, lb = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)), ub = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)))
#         # c32_1 = LinearConstraint(np.ones(len(initials)), lb = (opti_data['Last_Year_Period_Spends'].sum()*(1-0.2)), ub = (opti_data['Last_Year_Period_Spends'].sum()))
#         l_S32 = [c32]
#         # l_S32.append(c32_1)
#         cons = tuple(l_S32)
#         print(cons)
#     elif sub_strategy_name == "sub-strategy3":
#         c33_1 = NonlinearConstraint(Vol_func, lb = (-(opti_data['Last_Year_Period_NetSales'].sum())*(1+goal_perc_revenue)), ub = (-opti_data['Last_Year_Period_NetSales'].sum()))
#         c33_2 = LinearConstraint(np.ones(len(initials)), lb = goal_spend, ub = goal_spend)
#         l_S33 = [c33_1]
#         l_S33.append(c33_2)
#         cons = tuple(l_S33)
#         print(cons)
# # res = step_size_finder(strategy_name, sub_strategy_name, cons, bound, initials, activity_type, opti_data, goal_perc_revenue, goal_spend)

# COMMAND ----------

# ini_val = initials
# dat = opti_data
# strategy_name = strategy_name
# sub_strategy_name = sub_strategy_name
# opti_data = dat
# activity_type = activity_type
# cons = cons
# bound = bound
# initials = ini_val
# goal_perc_revenue = goal_perc_revenue
# goal_spend = goal_spend
# outputs = []
# # step_size_list = [[0.25, True],[0.25, False],[0.3, True],[0.35, False],[0.4, True],[0.4, False],[0.45, True],[0.45, False], [0.5, True],[0.5, False]]
# step_size_list = [[0.45,False],[0.45,True],[0.5,False],[0.5,True],[0.55,False],[0.55,True],[0.6,False],[0.6,True],[0.65,False],[0.65,True],[0.7,False],[0.7,True]]

# list_ = [(strategy_name, sub_strategy_name, cons, bound, initials, activity_type, dat, step) for step in step_size_list]
# # print("list",list_)
# with concurrent.futures.ProcessPoolExecutor(max_workers = len(step_size_list)) as executor:
#     for rs in executor.map(optim_details, list_):
#         # print(rs)
#         outputs.append(list(rs))
# # print(outputs)

# result_df = pd.DataFrame({"stepsize": step_size_list, "optimum": outputs})
# # print(result_df)
# step_size = []
# optimum = []
# grow = []
# spends = []
# for i in range(result_df.shape[0]):
#     opti_data['Optimized_Spends'] = result_df.iloc[i,1]
#     opti_data['Optimized_NS'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])*opti_data['PRICE']*opti_data['NS_TO_GPS']*opti_data['Adjustment_Factor']
#     # opti_data['Optimized_Volume'] = opti_data['BackCalc_Intercept']*(opti_data['Optimized_Spends']**opti_data['Spends_Coef'])
#     step_size.append(result_df.iloc[i,0])
#     optimum.append(result_df.iloc[i,1])
#     grow.append(((opti_data.Optimized_NS.sum() - opti_data.Last_Year_Period_NetSales.sum())/opti_data.Last_Year_Period_NetSales.sum()))
#     # grow.append(((opti_data.Optimized_Volume.sum() - opti_data.Last_Year_Period_Volume.sum())/opti_data.Last_Year_Period_Volume.sum()))
#     spends.append(((opti_data.Optimized_Spends.sum() - opti_data.Last_Year_Period_Spends.sum())/opti_data.Last_Year_Period_Spends.sum()))
#     opti_data = opti_data.drop(['Optimized_Spends', 'Optimized_NS'], axis = 1)
#     # opti_data.drop(['Optimized_Spends', 'Optimized_Volume'], axis = 1)

# df_all = pd.DataFrame({"Step": step_size, "Optimum": optimum, "Growth_percent": grow, "Spend_percent": spends})
# # if goal_spend == None:
# #     df_all = df_all[df_all['Spend_percent']<=0.005].reset_index(drop=True)
# #     spark.createDataFrame(df_all).display()
# #     df_all = df_all[(df_all['Growth_percent'] < 0) & (df_all['Spend_percent'] >= 0.01)].reset_index(drop=True) ### this is the new line

# # df_all = df_all.sort_values(by = 'Growth_percent', ascending = False)
# # # df_all[["Step", "Growth_percent", "Spend_percent"]].display()
# # print(df_all.iloc[0,0])
# # return df_all.iloc[0,1]

# COMMAND ----------

# print(goal_spend)

# COMMAND ----------

# if goal_spend == None:
#     df_all = df_all[df_all['Spend_percent']<=0.005].reset_index(drop=True)
#     df_all_new = df_all[(df_all['Growth_percent'] >= 0) | (df_all['Spend_percent'] < 0.01)].reset_index(drop=True) ### this is the new line
#     spark.createDataFrame(df_all_new).display()

# df_all = df_all.sort_values(by = 'Growth_percent', ascending = False)
# # df_all[["Step", "Growth_percent", "Spend_percent"]].display()
# print(df_all.iloc[0,0])

# COMMAND ----------

# DBTITLE 1,iter 1 modeling
opti_data["BackCalc_Intercept"] = opti_data["Last_Year_Period_Volume"] / (
    opti_data["Last_Year_Period_Spends"] ** opti_data["Spends_Coef"]
)
opti_data["BackCalc_Intercept"] = opti_data["BackCalc_Intercept"].clip(lower=0.0)
opti_data["Last_Year_Period_NetSales_Calc"] = (
    opti_data["BackCalc_Intercept"]
    * (opti_data["Last_Year_Period_Spends"] ** opti_data["Spends_Coef"])
    * opti_data["PRICE"]
    * opti_data["NS_TO_GPS"]
)
opti_data["Adjustment_Factor"] = (
    opti_data["Last_Year_Period_NetSales"] / opti_data["Last_Year_Period_NetSales_Calc"]
)
if (strategy_no == "3") and (sub_strategy_no == "1"):
    goal_spend = goal_spend - (non_opt_data.Last_Year_Period_Spends.sum())
    print("Revised goal spend = ", goal_spend)
    iter1_input_list[12] = goal_spend
    print(iter1_input_list)
lb_vec, ub_vec, list_os = optimizer_master(iter1_input_list, opti_data, "1")
out_data = opti_data.copy(True)
out_data = out_data.drop(
    [
        "Optimized_Spends",
        "Optimized_NS",
        "Optimized_Volume",
        "Optimized_GPS",
        # "Overall_Spends_change",
        "Spend_Change",
    ],
    axis=1,
)
out_data["Spends_lb"] = lb_vec * opti_data.loc[:, "Last_Year_Period_Spends"]
out_data["Spends_ub"] = ub_vec * opti_data.loc[:, "Last_Year_Period_Spends"]
out_data["Optimized_Spends"] = list_os
out_data["Spend_Change"] = (
    out_data["Optimized_Spends"] - out_data["Last_Year_Period_Spends"]
) / out_data["Last_Year_Period_Spends"]
out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume"] * (
    1 + out_data["Spend_Change"] * out_data["Spends_Coef"]
)  # out_data['Intercept']*(out_data['Optimized_Spends']**out_data['Spends_Coef'])
# out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
out_data["Optimized_GPS"] = (
    out_data["Optimized_Volume"]
    * out_data["Last_Year_Period_GPS"]
    / out_data["Last_Year_Period_Volume"]
)
out_data = out_data.fillna({"Optimized_Spends": 0, "Optimized_GPS": 0})
# out_data['Optimized_NetSales'] = out_data['Optimized_GPS']*out_data['NS_TO_GPS']
out_data["Optimized_NetSales"] = (
    out_data["Optimized_GPS"] - out_data["Optimized_Spends"]
)
out_data["Optimized_NetSales_Adj"] = (
    out_data["Optimized_NetSales"] * out_data["Adjustment_Factor"]
)
# out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends'])/out_data['Last_Year_Period_Spends']
out_data["Vol_Change"] = (
    out_data["Optimized_Volume"] - out_data["Last_Year_Period_Volume"]
) / out_data["Last_Year_Period_Volume"]

# if out_data["Spend_Change"]*out_data['Vol_Change']<0:
#     out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends"]*out_data["Spends_Coef"])
#     out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
#     out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume'])/out_data['Last_Year_Period_Volume']

out_data["NS_Change"] = (
    out_data["Optimized_NetSales"] - out_data["Last_Year_Period_NetSales"]
) / out_data["Last_Year_Period_NetSales"]
# out_data.display()
print("out_data created")


status_df = spark.createDataFrame(
    pd.DataFrame(
        {
            "SCENARIONAME": [scenario_name],
            "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
            "PROGRESSMESSAGE": [0.38],
        }
    )
)
# status_df.display()
data = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
    .load()
    .filter(sf.col("SCENARIONAME") != scenario_name)
)
sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
sf_output.write.format("snowflake").options(**options).option(
    "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
).mode("overwrite").save()

# COMMAND ----------

spark.createDataFrame(out_data).display()

# COMMAND ----------


# try:
    
    
#     opti_data['BackCalc_Intercept'] = opti_data['Last_Year_Period_Volume']/(opti_data['Last_Year_Period_Spends']**opti_data['Spends_Coef'])
#     opti_data['BackCalc_Intercept'] = opti_data['BackCalc_Intercept'].clip(lower=0.0)
#     opti_data['Last_Year_Period_NetSales_Calc'] = opti_data['BackCalc_Intercept']*(opti_data['Last_Year_Period_Spends']**opti_data['Spends_Coef'])*opti_data['PRICE']*opti_data['NS_TO_GPS']
#     opti_data['Adjustment_Factor'] = opti_data['Last_Year_Period_NetSales']/opti_data['Last_Year_Period_NetSales_Calc']
#     if (strategy_no == '3') and (sub_strategy_no == '1'):
#         goal_spend = goal_spend - (non_opt_data.Last_Year_Period_Spends.sum())
#         print("Revised goal spend = ", goal_spend)
#         iter1_input_list[12] = goal_spend
#         print(iter1_input_list)
#     lb_vec, ub_vec, list_os = optimizer_master(iter1_input_list, opti_data, '1')
#     out_data = opti_data.copy(True)
#     out_data = out_data.drop(['Optimized_Spends','Optimized_NS'], axis=1)
#     out_data['Spends_lb'] = lb_vec*opti_data.loc[:,'Last_Year_Period_Spends']
#     out_data['Spends_ub'] = ub_vec*opti_data.loc[:,'Last_Year_Period_Spends']
#     out_data['Optimized_Spends'] = list_os
#     out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends'])/out_data['Last_Year_Period_Spends']
#     out_data['Optimized_Volume'] = out_data["Last_Year_Period_Volume"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends"]*out_data["Spends_Coef"]) #out_data['Intercept']*(out_data['Optimized_Spends']**out_data['Spends_Coef'])
#     out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
#     out_data['Optimized_NetSales'] = out_data['Optimized_GPS']*out_data['NS_TO_GPS']
#     out_data['Optimized_NetSales_Adj'] = out_data['Optimized_NetSales']*out_data['Adjustment_Factor']
#     # out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends'])/out_data['Last_Year_Period_Spends']
#     out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume'])/out_data['Last_Year_Period_Volume']

#     # if out_data["Spend_Change"]*out_data['Vol_Change']<0:
#     #     out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends"]*out_data["Spends_Coef"])
#     #     out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
#     #     out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume'])/out_data['Last_Year_Period_Volume']

#     out_data['NS_Change'] = (out_data['Optimized_NetSales_Adj']-out_data['Last_Year_Period_NetSales'])/out_data['Last_Year_Period_NetSales']
#     # out_data.display()
#     print("out_data created")
    
    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.38]}))
#     # status_df.display()\
    
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.38]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    
    if non_opt_data.shape[0] != 0:
        non_opt_data['BackCalc_Intercept'] = [0]*non_opt_data.shape[0]
        non_opt_data['Last_Year_Period_NetSales_Calc'] = [0]*non_opt_data.shape[0]
        non_opt_data['Adjustment_Factor'] = [0]*non_opt_data.shape[0]
        non_opt_data['Spends_lb'] = 0.9*non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Spends_ub'] = 1.1*non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Optimized_Spends'] = non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Optimized_Volume'] = non_opt_data.loc[:,'Last_Year_Period_Volume']
        non_opt_data['Optimized_GPS'] = non_opt_data.loc[:,'Last_Year_Period_GPS']
        non_opt_data['Optimized_NetSales'] = non_opt_data.loc[:,'Last_Year_Period_NetSales']
        non_opt_data['Optimized_NetSales_Adj'] = non_opt_data['Optimized_NetSales']*non_opt_data['Adjustment_Factor']
        non_opt_data['Spend_Change'] = (non_opt_data['Optimized_Spends']-non_opt_data['Last_Year_Period_Spends'])/non_opt_data['Last_Year_Period_Spends']
        non_opt_data['Vol_Change'] = [0]*non_opt_data.shape[0]
        non_opt_data['NS_Change'] = [0]*non_opt_data.shape[0]
        non_opt_data = non_opt_data[out_data.columns]
        # non_opt_data.display()
        print("non_opt_data updated")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.40]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.40]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    
    if non_opt_data.shape[0] != 0:
        final_data = pd.concat([out_data,non_opt_data],axis=0)
    else:
        final_data = out_data.copy(True)
    
    # final_data.display()
    print("final_data created")
    print("Spend Change = ",(final_data.Optimized_Spends.sum()-final_data.Last_Year_Period_Spends.sum())/final_data.Last_Year_Period_Spends.sum())
    print("Volume Change = ",(final_data.Optimized_Volume.sum()-final_data.Last_Year_Period_Volume.sum())/final_data.Last_Year_Period_Volume.sum())
    print("Net Sales Change = ",(final_data.Optimized_NetSales_Adj.sum()-final_data.Last_Year_Period_NetSales.sum())/final_data.Last_Year_Period_NetSales.sum())

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.42]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.42]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:
    
    table_name = opt_database_name+'.'+first_iter_data
    print(table_name)
    spark.createDataFrame(final_data).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.44]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.44]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------



# %md
### Optimization at Product Level within Shisha x Customer Level

# COMMAND ----------

coef_info_df.display()

# COMMAND ----------

# data_to_opt = (coef_info_df
#                    .join(decomp_df,['SEGMENT','SHISHA','PLANNINGCUSTOMER'], 'left')
#                    .fillna(0, subset = ['SPENDS','VOLUME','GPS','NETSALES','PRICE'])
#                    .withColumnRenamed('SPENDS','Last_Year_Period_Spends_old')
#                    .withColumnRenamed('VOLUME','Last_Year_Period_Volume_old')
#                    .withColumnRenamed('GPS','Last_Year_Period_GPS_old')
#                    .withColumnRenamed('NETSALES','Last_Year_Period_NetSales_old')
#                    .withColumn('Total_LYPS',sf.sum('Last_Year_Period_Spends_old').over(w))
#                    .withColumn('Total_LYPV',sf.sum('Last_Year_Period_Volume_old').over(w))
#                    .withColumn('Total_LYPGPS',sf.sum('Last_Year_Period_GPS_old').over(w))
#                    .withColumn('Total_LYPNS',sf.sum('Last_Year_Period_NetSales_old').over(w))
#                    .withColumn('Spends_Perc_Contri',sf.when(sf.col('Total_LYPS')!=0,sf.col('Last_Year_Period_Spends_old')/sf.col('Total_LYPS')).otherwise(sf.lit(0).cast('double')))
#                    .withColumn('Volume_Perc_Contri',sf.when(sf.col('Total_LYPV')!=0,sf.col('Last_Year_Period_Volume_old')/sf.col('Total_LYPV')).otherwise(sf.lit(0).cast('double')))
#                    .withColumn('GPS_Perc_Contri',sf.when(sf.col('Total_LYPGPS')!=0,sf.col('Last_Year_Period_GPS_old')/sf.col('Total_LYPGPS')).otherwise(sf.lit(0).cast('double')))
#                    .withColumn('NS_Perc_Contri',sf.when(sf.col('Total_LYPNS')!=0,sf.col('Last_Year_Period_NetSales_old')/sf.col('Total_LYPNS')).otherwise(sf.lit(0).cast('double')))
#                    .drop('Total_LYPS','Total_LYPV','Total_LYPGPS','Total_LYPNS')
#                    .distinct()
#                    )
# data_to_opt.filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").display()

# COMMAND ----------

# DBTITLE 1,iter 2

try:
    
    
    decomp_df = spark.sql("select * from {0}.{1}".format(opt_database_name,"optimization_input"))
    coef_info_df = spark.sql("select * from {0}.{1}".format(opt_database_name,data_prep_iter2))
    # category_list = list(coef_info_df.select('CATEGORY').distinct().toPandas().CATEGORY.unique())
    # print(category_list)
    customer_list = list(coef_info_df.select('PLANNINGCUSTOMER').distinct().toPandas().PLANNINGCUSTOMER.unique())
    print(len(customer_list))

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.46]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.46]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------

decomp_df.filter("PLANNINGCUSTOMER == 'L5 JP NS Marks'").display()
spark.sql("select * from {0}.output_final_with_ns".format(decomp_database_name)).filter("YEAR == 2024 and MONTH >=7 and PLANNINGCUSTOMER == 'L5 JP NS Marks' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET'").display()

# COMMAND ----------

# DBTITLE 1,last yr same period calc
try:

    # Approach 1
    w = Window.partitionBy("SHISHA", "PLANNINGCUSTOMER")
    decomp_df = (
        decomp_df.withColumn(
            "STARTDATE",
            sf.to_date(
                sf.concat_ws("/", sf.lit("01"), sf.col("Month"), sf.col("Year")),
                "dd/MM/yyyy",
            ),
        )
        .filter(
            (sf.col("STARTDATE").isin(last_year_period))
            & (sf.col("MODELINGNAME").isin(spend_list))
            & (sf.col("BUSINESS") == business_type)
            & (sf.col("CATEGORY").isin(category_list))
            & (sf.col("CHANNEL").isin(channel_list))
            & (sf.col("SHISHA").isin(shisha_list))
            & (sf.col("PLANNINGCUSTOMER").isin(customer_list))
            & (sf.col("SPENDS") > 0)
            & (sf.col("VOLUME") >= 0)
        )
        .groupBy("SEGMENT", "SHISHA", "PLANNINGCUSTOMER")
        .agg(
            sf.sum("SPENDS").alias("SPENDS"),
            sf.sum("VOLUME").alias("VOLUME"),
            sf.sum("GPS").alias("GPS"),
            sf.sum("NETSALES").alias("NETSALES"),
            sf.mean("PRICE").alias("PRICE"),
        )
        .withColumn("SPENDS", sf.round(sf.col("SPENDS"), 3))
        .withColumn("VOLUME", sf.round(sf.col("VOLUME"), 3))
        .withColumn("GPS", sf.round(sf.col("GPS"), 3))
        .withColumn("NETSALES", sf.round(sf.col("NETSALES"), 3))
        .withColumn("PRICE", sf.round(sf.col("PRICE"), 3))
    )
    # decomp_df.display()
    print("Iter 2 decomp_df created")
    data_to_opt = (
        coef_info_df.join(decomp_df, ["SEGMENT", "SHISHA", "PLANNINGCUSTOMER"], "left")
        .fillna(0, subset=["SPENDS", "VOLUME", "GPS", "NETSALES", "PRICE"])
        .withColumnRenamed("SPENDS", "Last_Year_Period_Spends_old")
        .withColumnRenamed("VOLUME", "Last_Year_Period_Volume_old")
        .withColumnRenamed("GPS", "Last_Year_Period_GPS_old")
        .withColumnRenamed("NETSALES", "Last_Year_Period_NetSales_old")
        .withColumn("Total_LYPS", sf.sum("Last_Year_Period_Spends_old").over(w))
        .withColumn("Total_LYPV", sf.sum("Last_Year_Period_Volume_old").over(w))
        .withColumn("Total_LYPGPS", sf.sum("Last_Year_Period_GPS_old").over(w))
        .withColumn("Total_LYPNS", sf.sum("Last_Year_Period_NetSales_old").over(w))
        .withColumn(
            "Spends_Perc_Contri",
            sf.when(
                sf.col("Total_LYPS") != 0,
                sf.col("Last_Year_Period_Spends_old") / sf.col("Total_LYPS"),
            ).otherwise(sf.lit(0).cast("double")),
        )
        .withColumn(
            "Volume_Perc_Contri",
            sf.when(
                sf.col("Total_LYPV") != 0,
                sf.col("Last_Year_Period_Volume_old") / sf.col("Total_LYPV"),
            ).otherwise(sf.lit(0).cast("double")),
        )
        .withColumn(
            "GPS_Perc_Contri",
            sf.when(
                sf.col("Total_LYPGPS") != 0,
                sf.col("Last_Year_Period_GPS_old") / sf.col("Total_LYPGPS"),
            ).otherwise(sf.lit(0).cast("double")),
        )
        .withColumn(
            "NS_Perc_Contri",
            sf.when(
                sf.col("Total_LYPNS") != 0,
                sf.col("Last_Year_Period_NetSales_old") / sf.col("Total_LYPNS"),
            ).otherwise(sf.lit(0).cast("double")),
        )
        .drop("Total_LYPS", "Total_LYPV", "Total_LYPGPS", "Total_LYPNS")
        .distinct()
    )
    # data_to_opt.display()
    print("Iter 2 data_to_opt created")

    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
                "PROGRESSMESSAGE": [0.48],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()
except:
    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [2],
                "PROGRESSMESSAGE": [0.48],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()

# COMMAND ----------

data_to_opt.columns

# COMMAND ----------



try:
    
    year = int(last_year_period[0].split('-')[0])
    print(year)
    ec_info = (spark.sql("select * from {0}.mbt_wo_competition_price_corrected".format(decomp_database_name))
            #    .withColumn('BUSINESS',sf.when(sf.col('SEGMENT').contains('JP/C&B'),sf.lit('JP/Coffee&Beverage'))
            #                .otherwise('JP/Confectionery'))
            #    .withColumn('CATEGORY',sf.when(sf.col('SEGMENT').contains('JP/C&B/RSC'),sf.lit('JP/C&B/RSC'))
            #                .otherwise(sf.when(sf.col('SEGMENT').contains('JP/C&B/RTD'),sf.lit('JP/C&B/RTD'))
            #                           .otherwise(sf.when(sf.col('SEGMENT').contains('JP/C&B/SS'),sf.lit('JP/C&B/Single Serve'))
            #                                      .otherwise(sf.lit('JP/CNF/In To Home')))))
               )
    for item in range(len(spend_list)):
        if item == 0:
            ec_info = ec_info.withColumn('SPENDS',sf.col(spend_list[item]))
        else:
            ec_info = ec_info.withColumn('SPENDS',sf.col('SPENDS')+sf.col(spend_list[item])) 

    ec_info = (ec_info
               .withColumnRenamed('SHISHANM','SHISHA')
            #    .join((spark.sql("select * from {0}.seller_geo_map".format(decomp_database_name))
            #           .select('PLANNINGCUSTOMER',sf.col('CHANNELCD').alias('CHANNEL'))
            #           .distinct()
            #           ),['PLANNINGCUSTOMER'],'left')
            #    .filter((sf.col('YEAR') == year) & (sf.col('CHANNEL').isin(channel_list)) & (sf.col('BUSINESS')==business_type) & (sf.col('CATEGORY').isin(category_list)) & (sf.col('SHISHA').isin(shisha_list)) & (sf.col('PLANNINGCUSTOMER').isin(customer_list)) & (sf.col('SPENDS') > 0) & (sf.col('SALESQTYCASE') >= 0))
               .filter((sf.col('YEAR') == year) & (sf.col('PLANNINGCUSTOMER').isin(customer_list)) & (sf.col('SPENDS') > 0) & (sf.col('SALESQTYCASE') >= 0))
               .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
               .agg(sf.sum('SALESQTYCASE').alias('SALESQTYCASE'),
                    sf.sum('EC').alias('EC'))
               .withColumn('EC_PER_QTYCASE',sf.when(sf.col('SALESQTYCASE')!=0,sf.col('EC')/sf.col('SALESQTYCASE')).otherwise(sf.lit(0)))
               .select('SHISHA','PLANNINGCUSTOMER','SEGMENT','EC_PER_QTYCASE')
               .distinct()
               )          
    # ec_info.display()
    print("Iter 2 ec_info created")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.50]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.50]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------

(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data))
                         .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
                         .select('SHISHA','PLANNINGCUSTOMER',sf.col('Optimized_Spends').alias('Total_Spends_updt'),sf.col('Optimized_Volume').alias('Total_Volume_updt'),sf.col('Optimized_GPS_Adj').alias('Total_GPS_updt'),sf.col('Optimized_NetSales_Adj').alias('Total_NetSales_updt'))
                         .distinct()
                         ).display()

# COMMAND ----------


try:
    
    
    first_iter_output = (spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data))
                         .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
                         .select('SHISHA','PLANNINGCUSTOMER',sf.col('Optimized_Spends').alias('Total_Spends_updt'),sf.col('Optimized_Volume').alias('Total_Volume_updt'),sf.col('Optimized_GPS_Adj').alias('Total_GPS_updt'),sf.col('Optimized_NetSales_Adj').alias('Total_NetSales_updt'))
                         .distinct()
                         )
    data_to_opt = (data_to_opt
                   .join(first_iter_output, ['SHISHA','PLANNINGCUSTOMER'], 'left')
                   .distinct()
                   .withColumn('Last_Year_Period_Spends',sf.col('Spends_Perc_Contri')*sf.col('Total_Spends_updt'))
                   .withColumn('Last_Year_Period_Volume',sf.col('Volume_Perc_Contri')*sf.col('Total_Volume_updt'))
                #    .withColumn('Last_Year_Period_GPS',sf.col('GPS_Perc_Contri')*sf.col('Total_GPS_updt'))
                   .withColumn('Last_Year_Period_GPS',sf.col('Last_Year_Period_Volume')*sf.col('Last_Year_Period_GPS_old')/sf.col("Last_Year_Period_Volume_old"))
                   .fillna(0, subset=["Last_Year_Period_Spends","Last_Year_Period_GPS"])
                   .withColumn('Last_Year_Period_NetSales',sf.col('Last_Year_Period_GPS')-sf.col('Last_Year_Period_Spends'))
                   .drop('Total_Spends_updt','Total_Volume_updt','Total_GPS_updt','Total_NetSales_updt','Spends_Perc_Contri','Volume_Perc_Contri','GPS_Perc_Contri','NS_Perc_Contri')
                   .distinct()
                #    .withColumn('MC_PER_NETSALES',sf.when(sf.col('Last_Year_Period_NetSales')!=0,sf.round(sf.col('MC')/sf.col('Last_Year_Period_NetSales'),3)).otherwise(sf.lit(0).cast('double')))
                   .withColumn('NS_TO_GPS', sf.when(sf.col('Last_Year_Period_GPS')!=0,sf.round(sf.col('Last_Year_Period_NetSales')/sf.col('Last_Year_Period_GPS'),3)).otherwise(sf.lit(0)))
                   .withColumn('Key_Combination1', sf.concat(sf.col('SHISHA'),sf.lit('$'),sf.col('PLANNINGCUSTOMER')))
                   .join(ec_info,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'left')
                   .distinct()
                   )
    # data_to_opt.display()
    print("Iter 2 data_to_opt updated")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.52]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.52]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------

data_to_opt.columns

# COMMAND ----------


try:
    
    
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Last_Year_Period_Spends')).collect()[0][0])
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Last_Year_Period_Volume')).collect()[0][0])
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Last_Year_Period_NetSales')).collect()[0][0])
    print("***************************************************************************************************")
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Optimized_Spends')).collect()[0][0])
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Optimized_Volume')).collect()[0][0])
    print(spark.sql("select * from {0}.{1}".format(opt_database_name,first_iter_data)).select(sf.sum('Optimized_NetSales_Adj')).collect()[0][0])
    print("***************************************************************************************************")
    print(data_to_opt.select(sf.sum('Last_Year_Period_Spends')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_Volume')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_GPS')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_NetSales')).collect()[0][0])
    print("***************************************************************************************************")
    print(data_to_opt.select(sf.sum('Last_Year_Period_Spends_old')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_Volume_old')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_GPS_old')).collect()[0][0])
    print(data_to_opt.select(sf.sum('Last_Year_Period_NetSales_old')).collect()[0][0])
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.54]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.54]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------

(data_to_opt
                     .filter((sf.col('Last_Year_Period_Spends')>0) & (sf.col('Last_Year_Period_Volume')>0))
                     .select('Key_Combination', 'Key_Combination1', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 
                             'CLUSTER', 'CHANNEL', 'SHISHA', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'PRICE', 'EC_PER_QTYCASE', 'Last_Year_Period_Spends_old', 'Last_Year_Period_Volume_old', 'Last_Year_Period_GPS_old', 'Last_Year_Period_NetSales_old', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS', 'Intercept')
                     .toPandas()
                     ).display()

# COMMAND ----------


try:
    
    
    opti_data_all = (data_to_opt
                     .filter((sf.col('Last_Year_Period_Spends')>0) & (sf.col('Last_Year_Period_Volume')>0))
                     .select('Key_Combination', 'Key_Combination1', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 
                             'CLUSTER', 'CHANNEL', 'SHISHA', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'PRICE', 'EC_PER_QTYCASE', 'Last_Year_Period_Spends_old', 'Last_Year_Period_Volume_old', 'Last_Year_Period_GPS_old', 'Last_Year_Period_NetSales_old', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS', 'Intercept')
                     .toPandas()
                     )
    non_opt_data = (data_to_opt
                    .filter((sf.col('Last_Year_Period_Spends')>0) & (sf.col('Last_Year_Period_Volume')==0))
                    .select('Key_Combination', 'Key_Combination1', 'BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 
                             'CLUSTER', 'CHANNEL', 'SHISHA', 'PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'PRICE', 'EC_PER_QTYCASE', 'Last_Year_Period_Spends_old', 'Last_Year_Period_Volume_old', 'Last_Year_Period_GPS_old', 'Last_Year_Period_NetSales_old', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS')
                    .toPandas()
                    )
    # opti_data_all.display()
    print("Iter 2 opti_data_all created")
    if non_opt_data.shape[0] != 0:
    #     non_opt_data.display()
        print("Iter 2 non_opt_data created")

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.56]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.56]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------

iter2_input_list

# COMMAND ----------

spark.createDataFrame(opti_data).display()

# COMMAND ----------

# DBTITLE 1,iter 2 modeling debug
# opti_data_all['BackCalc_Intercept'] = opti_data_all['Last_Year_Period_Volume']/(opti_data_all['Last_Year_Period_Spends']**opti_data_all['Spends_Coef'])
# opti_data_all['BackCalc_Intercept'] = opti_data_all['BackCalc_Intercept'].clip(lower=0.0)
# count = 1
# opti_data_all['Last_Year_Period_NetSales_Calc'] = opti_data_all['BackCalc_Intercept']*(opti_data_all['Last_Year_Period_Spends']**opti_data_all['Spends_Coef'])*opti_data_all['PRICE']*opti_data_all['NS_TO_GPS']
# opti_data_all['Last_Year_Period_NetSales_old'] = opti_data_all['Last_Year_Period_GPS_old'] - opti_data_all['Last_Year_Period_Spends_old']
# opti_data_all['Adjustment_Factor'] = opti_data_all['Last_Year_Period_NetSales']/opti_data_all['Last_Year_Period_NetSales_Calc']
# # key = "TOKYO$L5 JP NS Belc Honbu"
# # key = "KYUSHU$L5 JP NS A COOP KagoshimakenHonbu"
# key = "KANSAI$L5 JP NS Sunplaza (KANSAI)"
# print(key)
# opti_data = opti_data_all[opti_data_all['Key_Combination1']==key].reset_index(drop=True)
# lb_vec, ub_vec, list_os = optimizer_master(iter2_input_list, opti_data, '2')
# out_data = opti_data.copy(True)
# out_data = out_data.drop(["Optimized_Spends", "Optimized_NS", "Optimized_Volume","Optimized_GPS", "Overall_Spends_change", "Spend_Change"], axis=1)
# out_data.columns

# COMMAND ----------

# DBTITLE 1,iter 2 modeling
opti_data_all["BackCalc_Intercept"] = opti_data_all["Last_Year_Period_Volume"] / (
    opti_data_all["Last_Year_Period_Spends"] ** opti_data_all["Spends_Coef"]
)
opti_data_all["BackCalc_Intercept"] = opti_data_all["BackCalc_Intercept"].clip(
    lower=0.0
)
count = 1
opti_data_all["Last_Year_Period_NetSales_Calc"] = (
    opti_data_all["BackCalc_Intercept"]
    * (opti_data_all["Last_Year_Period_Spends"] ** opti_data_all["Spends_Coef"])
    * opti_data_all["PRICE"]
    * opti_data_all["NS_TO_GPS"]
)
opti_data_all["Last_Year_Period_NetSales_old"] = (
    opti_data_all["Last_Year_Period_GPS_old"]
    - opti_data_all["Last_Year_Period_Spends_old"]
)
opti_data_all["Adjustment_Factor"] = (
    opti_data_all["Last_Year_Period_NetSales"]
    / opti_data_all["Last_Year_Period_NetSales_Calc"]
)
print(opti_data_all.Key_Combination1.nunique())
for key in list(opti_data_all.Key_Combination1.unique()):
    print(str(count) + " -> ", key)
    opti_data = opti_data_all[opti_data_all["Key_Combination1"] == key].reset_index(
        drop=True
    )
    lb_vec, ub_vec, list_os = optimizer_master(iter2_input_list, opti_data, "2")
    out_data = opti_data.copy(True)
    out_data = out_data.drop(
        [
            "Optimized_Spends",
            "Optimized_NS",
            "Optimized_Volume",
            "Optimized_GPS",
            # "Overall_Spends_change",
            "Spend_Change",
        ],
        axis=1,
    )
    out_data["Spends_lb"] = lb_vec * out_data.loc[:, "Last_Year_Period_Spends"]
    out_data["Spends_ub"] = ub_vec * out_data.loc[:, "Last_Year_Period_Spends"]
    out_data["Optimized_Spends"] = list_os
    out_data["Spend_Change"] = (
        out_data["Optimized_Spends"] - out_data["Last_Year_Period_Spends_old"]
    ) / out_data["Last_Year_Period_Spends_old"]
    out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume_old"] * (
        1 + out_data["Spend_Change"] * out_data["Spends_Coef"]
    )  # out_data['Intercept']*(out_data['Optimized_Spends']**out_data['Spends_Coef'])
    # out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
    out_data["Optimized_GPS"] = (
        out_data["Optimized_Volume"]
        * out_data["Last_Year_Period_GPS_old"]
        / out_data["Last_Year_Period_Volume_old"]
    )
    out_data = out_data.fillna({"Optimized_Spends": 0, "Optimized_GPS": 0})
    # out_data['Optimized_NetSales'] = out_data['Optimized_GPS']*out_data['NS_TO_GPS']
    out_data["Optimized_NetSales"] = (
        out_data["Optimized_GPS"] - out_data["Optimized_Spends"]
    )
    out_data["Optimized_NetSales_Adj"] = (
        out_data["Optimized_NetSales"] * out_data["Adjustment_Factor"]
    )
    out_data["EC"] = (
        out_data["Last_Year_Period_Volume_old"] * out_data["EC_PER_QTYCASE"]
    )
    out_data["MC"] = out_data["Last_Year_Period_NetSales_old"] - out_data["EC"]
    out_data["EC_new"] = out_data["Optimized_Volume"] * out_data["EC_PER_QTYCASE"]
    out_data["MC_new"] = out_data["Optimized_NetSales_Adj"] - out_data["EC_new"]
    # out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends_old'])/out_data['Last_Year_Period_Spends_old']
    out_data["Vol_Change"] = (
        out_data["Optimized_Volume"] - out_data["Last_Year_Period_Volume_old"]
    ) / out_data["Last_Year_Period_Volume_old"]

    # if out_data["Spend_Change"]*out_data['Vol_Change']<0:
    #     out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume_old"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends_old"]*out_data["Spends_Coef"])
    #     out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
    #     out_data['EC_new'] = out_data['Optimized_Volume']*out_data['EC_PER_QTYCASE']
    #     out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume_old'])/out_data['Last_Year_Period_Volume_old']

    out_data["NS_Change"] = (
        out_data["Optimized_NetSales"] - out_data["Last_Year_Period_NetSales_old"]
    ) / out_data["Last_Year_Period_NetSales_old"]
    print(
        "Spend Change = ",
        (out_data.Optimized_Spends.sum() - out_data.Last_Year_Period_Spends_old.sum())
        / out_data.Last_Year_Period_Spends_old.sum(),
    )
    print(
        "Vol Change = ",
        (out_data.Optimized_Volume.sum() - out_data.Last_Year_Period_Volume_old.sum())
        / out_data.Last_Year_Period_Volume_old.sum(),
    )
    print(
        "NS Change = ",
        (
            out_data.Optimized_NetSales_Adj.sum()
            - out_data.Last_Year_Period_NetSales_old.sum()
        )
        / out_data.Last_Year_Period_NetSales_old.sum(),
    )
    if count == 1:
        table_name = opt_database_name + "." + second_iter_data
        print(table_name)
        spark.createDataFrame(out_data).write.mode("overwrite").option(
            "overwriteSchema", True
        ).saveAsTable(table_name)
    else:
        spark.createDataFrame(out_data).write.mode("append").option(
            "overwriteSchema", True
        ).saveAsTable(table_name)
    count = count + 1
print("Iter 2 out_data_all created")


status_df = spark.createDataFrame(
    pd.DataFrame(
        {
            "SCENARIONAME": [scenario_name],
            "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
            "PROGRESSMESSAGE": [0.58],
        }
    )
)
# status_df.display()
data = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
    .load()
    .filter(sf.col("SCENARIONAME") != scenario_name)
)
sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
sf_output.write.format("snowflake").options(**options).option(
    "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
).mode("overwrite").save()


spark.createDataFrame(out_data).groupBy("SHISHA", "PLANNINGCUSTOMER").agg(
    sf.sum("Last_Year_Period_Spends").alias("Last_Year_Period_Spends"),
    sf.sum("Optimized_Spends").alias("Optimized_Spends"),
).filter(
    sf.round(sf.col("Optimized_Spends"), 0)
    != sf.round(sf.col("Last_Year_Period_Spends"), 0)
).display()

# COMMAND ----------

# spark.createDataFrame(out_data).display()

# COMMAND ----------

# try:
    
#     opti_data_all['BackCalc_Intercept'] = opti_data_all['Last_Year_Period_Volume']/(opti_data_all['Last_Year_Period_Spends']**opti_data_all['Spends_Coef']) Last_Year_Period_Volume_old
#     opti_data_all['BackCalc_Intercept'] = opti_data_all['BackCalc_Intercept'].clip(lower=0.0)
#     count = 1
#     opti_data_all['Last_Year_Period_NetSales_Calc'] = opti_data_all['BackCalc_Intercept']*(opti_data_all['Last_Year_Period_Spends']**opti_data_all['Spends_Coef'])*opti_data_all['PRICE']*opti_data_all['NS_TO_GPS']
#     opti_data_all['Adjustment_Factor'] = opti_data_all['Last_Year_Period_NetSales']/opti_data_all['Last_Year_Period_NetSales_Calc']
#     print(opti_data_all.Key_Combination1.nunique())
#     for key in list(opti_data_all.Key_Combination1.unique()):
#         print(str(count)+" -> ", key)
#         opti_data = opti_data_all[opti_data_all['Key_Combination1']==key].reset_index(drop=True)
#         lb_vec, ub_vec, list_os = optimizer_master(iter2_input_list, opti_data, '2')
#         out_data = opti_data.copy(True)
#         out_data = out_data.drop(['Optimized_Spends','Optimized_NS'], axis=1)
#         out_data['Spends_lb'] = lb_vec*out_data.loc[:,'Last_Year_Period_Spends']
#         out_data['Spends_ub'] = ub_vec*out_data.loc[:,'Last_Year_Period_Spends']
#         out_data['Optimized_Spends'] = list_os
#         out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends_old'])/out_data['Last_Year_Period_Spends_old']
#         out_data['Optimized_Volume'] = out_data["Last_Year_Period_Volume"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends"]*out_data["Spends_Coef"]) # out_data['Intercept']*(out_data['Optimized_Spends']**out_data['Spends_Coef'])
#         out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
#         out_data['Optimized_NetSales'] = out_data['Optimized_GPS']*out_data['NS_TO_GPS']
#         out_data['Optimized_NetSales_Adj'] = out_data['Optimized_NetSales']*out_data['Adjustment_Factor']
#         out_data['EC'] = out_data['Last_Year_Period_Volume_old']*out_data['EC_PER_QTYCASE']
#         out_data['MC'] = out_data['Last_Year_Period_NetSales_old'] - out_data['EC']
#         out_data['EC_new'] = out_data['Optimized_Volume']*out_data['EC_PER_QTYCASE']
#         out_data['MC_new'] = out_data['Optimized_NetSales_Adj'] - out_data['EC_new']
#         # out_data['Spend_Change'] = (out_data['Optimized_Spends']-out_data['Last_Year_Period_Spends_old'])/out_data['Last_Year_Period_Spends_old']
#         out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume_old'])/out_data['Last_Year_Period_Volume_old']

#         # if out_data["Spend_Change"]*out_data['Vol_Change']<0:
#         #     out_data["Optimized_Volume"] = out_data["Last_Year_Period_Volume_old"]*(1+out_data["Spend_Change"]/out_data["Last_Year_Period_Spends_old"]*out_data["Spends_Coef"])
#         #     out_data['Optimized_GPS'] = out_data['Optimized_Volume']*out_data['PRICE']
#         #     out_data['EC_new'] = out_data['Optimized_Volume']*out_data['EC_PER_QTYCASE']
#         #     out_data['Vol_Change'] = (out_data['Optimized_Volume']-out_data['Last_Year_Period_Volume_old'])/out_data['Last_Year_Period_Volume_old']

#         out_data['NS_Change'] = (out_data['Optimized_NetSales_Adj']-out_data['Last_Year_Period_NetSales_old'])/out_data['Last_Year_Period_NetSales_old']
#         print("Spend Change = ", (out_data.Optimized_Spends.sum() - out_data.Last_Year_Period_Spends_old.sum())/out_data.Last_Year_Period_Spends_old.sum())
#         print("Vol Change = ", (out_data.Optimized_Volume.sum() - out_data.Last_Year_Period_Volume_old.sum())/out_data.Last_Year_Period_Volume_old.sum())
#         print("NS Change = ", (out_data.Optimized_NetSales_Adj.sum() - out_data.Last_Year_Period_NetSales_old.sum())/out_data.Last_Year_Period_NetSales_old.sum())
#         if count == 1:
#             table_name = opt_database_name+'.'+second_iter_data
#             print(table_name)
#             spark.createDataFrame(out_data).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)
#         else:
#             spark.createDataFrame(out_data).write.mode("append").option("overwriteSchema", True).saveAsTable(table_name)
#         count = count+1
#     print("Iter 2 out_data_all created")
    
    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.58]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.58]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------





# spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)).display()

# COMMAND ----------


try:
    
    if non_opt_data.shape[0] != 0:
        non_opt_data['BackCalc_Intercept'] = [0]*non_opt_data.shape[0]
        non_opt_data['Last_Year_Period_NetSales_Calc'] = [0]*non_opt_data.shape[0]
        non_opt_data['Adjustment_Factor'] = [0]*non_opt_data.shape[0]
        non_opt_data['Spends_lb'] = 0.9*non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Spends_ub'] = 1.1*non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Optimized_Spends'] = non_opt_data.loc[:,'Last_Year_Period_Spends']
        non_opt_data['Optimized_Volume'] = non_opt_data.loc[:,'Last_Year_Period_Volume']
        non_opt_data['Optimized_GPS'] = non_opt_data.loc[:,'Last_Year_Period_GPS']
        non_opt_data['Optimized_NetSales'] = non_opt_data.loc[:,'Last_Year_Period_NetSales']
        non_opt_data['Optimized_NetSales_Adj'] = non_opt_data['Optimized_NetSales']*non_opt_data['Adjustment_Factor']
        non_opt_data['EC'] = non_opt_data['Last_Year_Period_Volume_old']*non_opt_data['EC_PER_QTYCASE']
        non_opt_data['MC'] = non_opt_data['Last_Year_Period_NetSales_old']-non_opt_data['EC']
        non_opt_data['EC_new'] = non_opt_data['Optimized_Volume']*non_opt_data['EC_PER_QTYCASE']
        non_opt_data['MC_new'] = non_opt_data['Optimized_NetSales_Adj']-non_opt_data['EC_new']
        non_opt_data['Spend_Change'] = (non_opt_data['Optimized_Spends']-non_opt_data['Last_Year_Period_Spends_old'])/non_opt_data['Last_Year_Period_Spends_old']
        non_opt_data['Vol_Change'] = [0]*non_opt_data.shape[0]
        non_opt_data['NS_Change'] = [0]*non_opt_data.shape[0]
        non_opt_data = non_opt_data[out_data.columns]
        # non_opt_data.display()
        print("Iter 2 non_opt_data updated")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.60]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.60]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------


try:
    
    
    if non_opt_data.shape[0] != 0:
        table_name = opt_database_name+'.'+second_iter_data
        print(table_name)
        (spark.createDataFrame(non_opt_data)
         .withColumn('BackCalc_Intercept', sf.col('BackCalc_Intercept').cast('double'))
         .withColumn('Last_Year_Period_NetSales_Calc', sf.col('Last_Year_Period_NetSales_Calc').cast('double'))
         .withColumn('Adjustment_Factor', sf.col('Adjustment_Factor').cast('double'))
         .withColumn('Vol_Change', sf.col('Vol_Change').cast('double'))
         .withColumn('NS_Change', sf.col('NS_Change').cast('double'))
         .write.mode("append").option("overwriteSchema", True).saveAsTable(table_name))
    print("Iter 2 optimization output stored")

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.62]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.62]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------

spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)).groupBy("SHISHA", "PLANNINGCUSTOMER").agg(sf.sum("Last_Year_Period_Spends").alias("Last_Year_Period_Spends"), sf.sum("Optimized_Spends").alias("Optimized_Spends")).withColumn("change_perc", (sf.col("Optimized_Spends")/sf.col("Last_Year_Period_Spends")-1)*100).filter(sf.round(sf.col("Optimized_Spends"), 0) != sf.round(sf.col("Last_Year_Period_Spends"), 0)).display()


# COMMAND ----------


try:
    
    
    # Final summary
    final_opt_data = spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data))
    print("Spend Change = ", (final_opt_data.select(sf.sum('Optimized_Spends')).collect()[0][0]-final_opt_data.select(sf.sum('Last_Year_Period_Spends_old')).collect()[0][0])/final_opt_data.select(sf.sum('Last_Year_Period_Spends_old')).collect()[0][0])
    print("Volume Change = ", (final_opt_data.select(sf.sum('Optimized_Volume')).collect()[0][0]-final_opt_data.select(sf.sum('Last_Year_Period_Volume_old')).collect()[0][0])/final_opt_data.select(sf.sum('Last_Year_Period_Volume_old')).collect()[0][0])
    # print("GPS Change = ", (final_opt_data.select(sf.sum('Optimized_GPS')).collect()[0][0]-final_opt_data.select(sf.sum('Last_Year_Period_GPS')).collect()[0][0])/final_opt_data.select(sf.sum('Last_Year_Period_GPS')).collect()[0][0])
    print("NS Change = ", (final_opt_data.select(sf.sum('Optimized_NetSales_Adj')).collect()[0][0]-final_opt_data.select(sf.sum('Last_Year_Period_NetSales_old')).collect()[0][0])/final_opt_data.select(sf.sum('Last_Year_Period_NetSales_old')).collect()[0][0])

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.64]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.64]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------



# %md
### HO Branch Modification

# COMMAND ----------

# spark.read.format("snowflake").options(**options).option("dbtable","VW_HO_BRANCH_CUSTOMER_CODE_MAP").load().display()

# COMMAND ----------

# spark.sql("select * from {0}.ho_branch_info".format(opt_database_name)).select("OLD_PLANNINGCUSTOMER").distinct().display()

# COMMAND ----------


try:
    
    if spend_list == ['PRICEOFF']:
        df = (spark.sql("select * from {0}.ho_branch_info".format(opt_database_name))
              .withColumnRenamed('NEW_PLANNINGCUSTOMER','PLANNINGCUSTOMER')
              .withColumnRenamed('SHISHANM','SHISHA')
              .withColumn('MONTHSTART',sf.concat(sf.col('YEAR'),sf.lit('-'),sf.col('MONTH'),sf.lit('-01')))
              .filter(sf.col('MONTHSTART').isin(last_year_period))
              .groupBy('SEGMENT','SHISHA','PLANNINGCUSTOMER','OLD_PLANNINGCUSTOMER')
              .agg(sf.sum('INI_PRICEOFF').alias('INI_PRICEOFF'),
                   sf.sum('ADD_PRICEOFF').alias('ADD_PRICEOFF'))
              .withColumn('TOTAL',sf.col('INI_PRICEOFF')+sf.col('ADD_PRICEOFF'))
              .withColumn('INI_CONTRI_PO',sf.when(sf.col('TOTAL')!=0,sf.round(sf.col('INI_PRICEOFF')/sf.col('TOTAL'),3)).otherwise(sf.lit(0)))
              .withColumn('ADD_CONTRI_PO',sf.when(sf.col('TOTAL')!=0,sf.round(sf.col('ADD_PRICEOFF')/sf.col('TOTAL'),3)).otherwise(sf.lit(0)))
              .drop('INI_PRICEOFF','ADD_PRICEOFF','TOTAL')
              .join(spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)),
                    ['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'right')
              .distinct()
              ).toPandas()
    elif spend_list == ['PROMOTIONAD']:
        df = (spark.sql("select * from {0}.ho_branch_info".format(opt_database_name))
              .withColumnRenamed('NEW_PLANNINGCUSTOMER','PLANNINGCUSTOMER')
              .withColumnRenamed('SHISHANM','SHISHA')
              .withColumn('MONTHSTART',sf.concat(sf.col('YEAR'),sf.lit('-'),sf.col('MONTH'),sf.lit('-01')))
              .filter(sf.col('MONTHSTART').isin(last_year_period))
              .groupBy('SEGMENT','SHISHA','PLANNINGCUSTOMER','OLD_PLANNINGCUSTOMER')
              .agg(sf.sum('INI_PROMOTIONAD').alias('INI_PROMOTIONAD'),
                   sf.sum('ADD_PROMOTIONAD').alias('ADD_PROMOTIONAD'))
              .withColumn('TOTAL',sf.col('INI_PROMOTIONAD')+sf.col('ADD_PROMOTIONAD'))
              .withColumn('INI_CONTRI_PA',sf.when(sf.col('TOTAL')!=0,sf.round(sf.col('INI_PROMOTIONAD')/sf.col('TOTAL'),3)).otherwise(sf.lit(0)))
              .withColumn('ADD_CONTRI_PA',sf.when(sf.col('TOTAL')!=0,sf.round(sf.col('ADD_PROMOTIONAD')/sf.col('TOTAL'),3)).otherwise(sf.lit(0)))
              .drop('INI_PROMOTIONAD','ADD_PROMOTIONAD','TOTAL')
              .join(spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)),
                    ['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'right')
              .distinct()
              ).toPandas()
    # df.display()
    print("df created")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.66]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.66]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------



# mapping_data = (spark.createDataFrame(df)
#                 .groupBy('OLD_PLANNINGCUSTOMER')
#                 .agg(sf.collect_set('PLANNINGCUSTOMER').alias('BRANCH_LIST'))
#                 .filter(sf.size(sf.col('BRANCH_LIST'))>1)
#                 .withColumn('BRANCH_LIST', sf.col('BRANCH_LIST').cast('string'))
#                 .withColumn('SCENARIO_NAME',sf.lit(scenario_name))
#                 .withColumn('SECTION_NAME',sf.lit(None))
#                 )
# # mapping_data.display()
# mapping_df = spark.read.format("snowflake").options(**options).option("dbtable", "MID_JP_FF_OPTOUTPUT_HO_BRANCH_MAP").load().filter(sf.col('SCENARIO_NAME')!=scenario_name)
# out_map = mapping_df.unionByName(mapping_data, allowMissingColumns=True).distinct()
# out_map.write.format("snowflake").options(**options).option("dbtable", "MID_JP_FF_OPTOUTPUT_HO_BRANCH_MAP").mode("overwrite").save()

# spark.read.format("snowflake").options(**options).option("dbtable", "MID_JP_FF_OPTOUTPUT_HO_BRANCH_MAP").load().filter(sf.col('SCENARIO_NAME')==scenario_name).show()
# print("HO Branch mapping created")

# COMMAND ----------




# l = mapping_data.toPandas().iloc[0,1]
# l = l[1:(len(l)-1)].split(",")
# print(l)

# COMMAND ----------



try:
    
    
    if spend_list == ['PRICEOFF']:
        value_variables = ['INI_CONTRI_PO', 'ADD_CONTRI_PO']
    elif spend_list == ['PROMOTIONAD']:
        value_variables = ['INI_CONTRI_PA', 'ADD_CONTRI_PA']

    df1 = (spark.createDataFrame(pd.melt(df,
                                        id_vars=['Key_Combination', 'Key_Combination1', 'BUSINESS', 'CATEGORY', 
                                                 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER', 'MODELINGNAME', 'Spends_Coef', 'PRICE', 'EC_PER_QTYCASE', 'Last_Year_Period_Spends_old', 'Last_Year_Period_Volume_old', 'Last_Year_Period_GPS_old', 'Last_Year_Period_NetSales_old', 'Last_Year_Period_Spends', 'Last_Year_Period_Volume', 'Last_Year_Period_GPS', 'Last_Year_Period_NetSales', 'NS_TO_GPS', 'BackCalc_Intercept', 'Last_Year_Period_NetSales_Calc', 'Adjustment_Factor', 'Spends_lb', 'Spends_ub', 'Optimized_Spends', 'Optimized_Volume', 'Optimized_GPS', 'Optimized_NetSales', 'Optimized_NetSales_Adj', 'EC', 'MC', 'EC_new', 'MC_new', 'Spend_Change', 'Vol_Change', 'NS_Change'], 
                                        value_vars = value_variables, 
                                        var_name='CONTRI_TYPE',
                                        value_name='CONTRI_VAL')
                                )
           .distinct()
           )
    print(df1.select('Key_Combination').distinct().count())
    df1_g1 = (df1
              .groupBy('Key_Combination')
              .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
              .filter("CONTRI_VAL = 0")
              .drop('CONTRI_VAL')
              .join(spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)),'Key_Combination','inner')
              )
    print(df1_g1.select('Key_Combination').distinct().count())
    df1_g2 = (df1
              .groupBy('Key_Combination')
              .agg(sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
              .filter("CONTRI_VAL != 0")
              .drop('CONTRI_VAL')
              .join(df1,'Key_Combination','left')
              .filter(sf.col('CONTRI_VAL')!=0)
              )
    print(df1_g2.select('Key_Combination').distinct().count())
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.68]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.68]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    
    print(df1_g2.count())
    print((df1_g2
           .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
           .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
           .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
           .filter(sf.col('CONTRI_VAL')<2)
           .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
           .distinct()
           .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
           ).count())
    if spend_list == ['PRICEOFF']:
         print((df1_g2
               .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
               .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                    sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                    sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
               .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO"))))
               .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
               .distinct()
               .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
               .distinct()
               ).count())
         print((df1_g2
                .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO")))))
                .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                .distinct()
                .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                .distinct()
                ).count())
    elif spend_list == ['PROMOTIONAD']:
         print((df1_g2
               .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
               .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                    sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                    sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
               .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
               .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA"))))
               .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
               .distinct()
               .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
               .distinct()
               ).count())
         print((df1_g2
                .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                     sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                     sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA")))))
                .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                .distinct()
                .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                .distinct()
                ).count())

    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.70]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.70]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------


try:
    
    
    
    group_one = (df1_g2
                 .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                 .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                      sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                      sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                 .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                 .filter(sf.col('CONTRI_VAL')<2)
                 .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                 .distinct()
                 .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                 )
    # print(group_one.count())
    if spend_list == ['PRICEOFF']:
        group_two = (df1_g2
                     .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                     .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                          sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                          sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                     .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                     .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO"))))
                     .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                     .distinct()
                     .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                     .filter((sf.col('CONTRI_VAL')!=1))
                     .distinct()
                     )
        # group_two.filter(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER')).display()
        # group_two.groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()
        # print(group_two.count())
        w = Window.partitionBy('Key_Combination')
        group_three = (df1_g2
                       .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                       .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                            sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                            sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                       .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                       .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PO"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PO")))))
                       .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                       .distinct()
                       .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                       .withColumn('COUNT',sf.count('Key_Combination').over(w))
                       .withColumn('CONTRI_VAL',sf.col('CONTRI_VAL')/sf.col('COUNT'))
                       .drop('COUNT')
                       .distinct()
                       )
    elif spend_list == ['PROMOTIONAD']:
        group_two = (df1_g2
                     .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                     .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                          sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                          sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                     .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                     .filter((sf.col('CONTRI_VAL')==2)&((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA"))))
                     .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                     .distinct()
                     .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                     .filter((sf.col('CONTRI_VAL')!=1))
                     .distinct()
                     )
        # group_two.filter(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER')).display()
        # group_two.groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT').agg(sf.collect_list('CONTRI_TYPE'),sf.sum('CONTRI_VAL')).display()
        # print(group_two.count())
        w = Window.partitionBy('Key_Combination')
        group_three = (df1_g2
                       .groupBy('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                       .agg(sf.collect_list('CONTRI_TYPE').alias('CONTRI_LIST'),
                            sf.collect_set('CONTRI_TYPE').alias('CONTRI_SET'),
                            sf.sum('CONTRI_VAL').alias('CONTRI_VAL'))
                       .withColumn('CONTRI_VAL',sf.round(sf.col('CONTRI_VAL'),3))
                       .filter((sf.col('CONTRI_VAL')==2)&(~((sf.array_contains(sf.col('CONTRI_SET'),"ADD_CONTRI_PA"))&(sf.array_contains(sf.col('CONTRI_SET'),"INI_CONTRI_PA")))))
                       .select('SHISHA','PLANNINGCUSTOMER','SEGMENT')
                       .distinct()
                       .join(df1_g2,['SHISHA','PLANNINGCUSTOMER','SEGMENT'],'inner')
                       .withColumn('COUNT',sf.count('Key_Combination').over(w))
                       .withColumn('CONTRI_VAL',sf.col('CONTRI_VAL')/sf.col('COUNT'))
                       .drop('COUNT')
                       .distinct()
                       )
    opt_updt = group_one.unionByName(group_two).distinct().unionByName(group_three).distinct()
    columns = ['Last_Year_Period_Spends_old', 'Last_Year_Period_Volume_old', 'Last_Year_Period_GPS_old', 'Last_Year_Period_NetSales_old', 'Optimized_Spends', 'Optimized_Volume', 'Optimized_GPS', 'Optimized_NetSales_Adj']
    for col in columns:
        opt_updt = opt_updt.withColumn(col,sf.round(sf.col(col)*sf.col('CONTRI_VAL'),3))

    if spend_list == ['PRICEOFF']:
        opt_updt = (opt_updt
                    .withColumn('PLANNINGCUSTOMER_final',sf.when(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER'),sf.col('PLANNINGCUSTOMER'))
                                .otherwise(sf.when((sf.col('PLANNINGCUSTOMER')!=sf.col('OLD_PLANNINGCUSTOMER'))&(sf.col('CONTRI_TYPE')=='INI_CONTRI_PO'),sf.col('PLANNINGCUSTOMER'))
                                           .otherwise(sf.col('OLD_PLANNINGCUSTOMER'))))
                    )
    elif spend_list == ['PROMOTIONAD']:
        opt_updt = (opt_updt
                    .withColumn('PLANNINGCUSTOMER_final',sf.when(sf.col('PLANNINGCUSTOMER')==sf.col('OLD_PLANNINGCUSTOMER'),sf.col('PLANNINGCUSTOMER'))
                                .otherwise(sf.when((sf.col('PLANNINGCUSTOMER')!=sf.col('OLD_PLANNINGCUSTOMER'))&(sf.col('CONTRI_TYPE')=='INI_CONTRI_PA'),sf.col('PLANNINGCUSTOMER'))
                                           .otherwise(sf.col('OLD_PLANNINGCUSTOMER'))))
                    )
        
    opt_updt = (opt_updt
                .drop('PLANNINGCUSTOMER', 'OLD_PLANNINGCUSTOMER')
                .withColumnRenamed('PLANNINGCUSTOMER_final','PLANNINGCUSTOMER')
                .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
                .groupBy('BUSINESS', 'CATEGORY', 'SUBCATEGORY', 'SEGMENT', 'SHISHA', 'CLUSTER', 'CHANNEL', 'PLANNINGCUSTOMER')
                .agg(sf.mean('EC_PER_QTYCASE').alias('EC_PER_QTYCASE'),
                     sf.mean('PRICE').alias('PRICE'),
                     sf.sum('Last_Year_Period_Spends').alias('Last_Year_Period_Spends_old'),
                     sf.sum('Last_Year_Period_Volume').alias('Last_Year_Period_Volume_old'),
                     sf.sum('Last_Year_Period_GPS').alias('Last_Year_Period_GPS_old'),
                     sf.sum('Last_Year_Period_NetSales').alias('Last_Year_Period_NetSales_old'),
                     sf.sum('Optimized_Spends').alias('Optimized_Spends'),
                     sf.sum('Optimized_Volume').alias('Optimized_Volume'),
                     sf.sum('Optimized_GPS_Adj').alias('Optimized_GPS_Adj'),
                     sf.sum('Optimized_NetSales_Adj').alias('Optimized_NetSales_Adj'))
                .withColumn('EC',sf.col('EC_PER_QTYCASE')*sf.col('Last_Year_Period_Volume_old'))
                .withColumn('MC',sf.col('Last_Year_Period_NetSales_old')-sf.col('EC'))
                .withColumn('EC_new',sf.col('EC_PER_QTYCASE')*sf.col('Optimized_Volume'))
                .withColumn('MC_new',sf.col('Optimized_NetSales_Adj')-sf.col('EC_new'))
                .withColumn('Spend_Change',
                            sf.when(sf.col('Last_Year_Period_Spends_old')!=0,sf.round((sf.col('Optimized_Spends')-sf.col('Last_Year_Period_Spends_old'))/sf.col('Last_Year_Period_Spends_old'),3))
                            .otherwise(sf.lit(0)))
                .withColumn('Vol_Change',
                            sf.when(sf.col('Last_Year_Period_Volume_old')!=0,sf.round((sf.col('Optimized_Volume')-sf.col('Last_Year_Period_Volume_old'))/sf.col('Last_Year_Period_Volume_old'),3))
                            .otherwise(sf.lit(0)))
                .withColumn('NS_Change',
                            sf.when(sf.col('Last_Year_Period_NetSales_old')!=0,sf.round((sf.col('Optimized_NetSales_Adj')-sf.col('Last_Year_Period_NetSales_old'))/sf.col('Last_Year_Period_NetSales_old'),3))
                            .otherwise(sf.lit(0)))
                )
    # opt_updt.display()
    print("opt_updt created")
    
    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.72]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.72]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()





# COMMAND ----------


try:
    
    opt_updt = (opt_updt
                .unionByName((df1_g1
                              .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
                              .select(*opt_updt.columns)
                              ))
                )
    print(opt_updt.count())

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.74]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.74]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------


try:
    
    opt_updt.select(sf.sum('Last_Year_Period_Spends_old'),sf.sum('Last_Year_Period_Volume_old'),sf.sum('Last_Year_Period_GPS_old'),sf.sum('Last_Year_Period_NetSales_old'),sf.sum('Optimized_Spends'),sf.sum('Optimized_Volume'),sf.sum('Optimized_GPS_Adj'),sf.sum('Optimized_NetSales_Adj')).show()
    spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data)).withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor')).select(sf.sum('Last_Year_Period_Spends_old'),sf.sum('Last_Year_Period_Volume_old'),sf.sum('Last_Year_Period_GPS_old'),sf.sum('Last_Year_Period_NetSales_old'),sf.sum('Optimized_Spends'),sf.sum('Optimized_Volume'),sf.sum('Optimized_GPS_Adj'),sf.sum('Optimized_NetSales_Adj')).show()

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.76]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.76]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------



try:
    
    table_name = opt_database_name+'.'+second_iter_data+'_updt'
    print(table_name)
    opt_updt.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

    
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.78]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.78]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()




# COMMAND ----------

opt_database_name, channel_list

# COMMAND ----------

# DBTITLE 1,decomp_0_0
decomp_df_0_0 = spark.sql("select * from {0}.optimization_input".format(opt_database_name)).withColumn(
    "STARTDATE",
    sf.to_date(
        sf.concat_ws("/", sf.lit("01"), sf.col("MONTH"), sf.col("YEAR")), "dd/MM/yyyy"
    ),
).filter(
    (sf.col("STARTDATE").isin(last_year_period))
    & (sf.col("MODELINGNAME").isin(spend_list))
    & (sf.col("BUSINESS") == business_type)
    & (sf.col("CATEGORY").isin(category_list))
    & (sf.col("CHANNEL").isin(channel_list))
    & (sf.col("SHISHA").isin(shisha_list))
    & (sf.col("SPENDS") > 0)
    & (sf.col("VOLUME") >= 0)
)
decomp_df_0_0.display()

decomp_df_0_0 = (
    decomp_df_0_0.groupBy(
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "SEGMENT",
        "SHISHA",
        "CLUSTER",
        "CHANNEL",
        "PLANNINGCUSTOMER",
        "MODELINGNAME",
    )
    .agg(
        sf.sum("SPENDS").alias("SPENDS"),
        sf.sum("GPS").alias("GPS"),
        sf.sum("NETSALES").alias("NETSALES"),
        sf.sum("VOLUME").alias("VOLUME"),
        sf.avg("PRICE").alias("PRICE"),
        sf.sum("EC").alias("EC"),
        sf.sum("EC_PER_QTYCASE").alias("EC_PER_QTYCASE"),
        sf.sum("MC").alias("MC"),
    )
    .withColumn("SCENARIO_NAME", sf.lit(scenario_name))
    .withColumn("DATE", sf.lit(input_params["optimizationInput"]["max"]))
)

# COMMAND ----------

opti_output = (
    spark.sql(
        "select * from {0}.{1}".format(opt_database_name, second_iter_data + "_updt")
    )
    #    .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
    .select(
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "SEGMENT",
        "SHISHA",
        "CLUSTER",
        "CHANNEL",
        "PLANNINGCUSTOMER",
        sf.col("Optimized_Spends").alias("SPENDS"),
        sf.col("Optimized_GPS_Adj").alias("GPS"),
        sf.col("Optimized_NetSales_Adj").alias("NETSALES"),
        sf.col("Optimized_Volume").alias("VOLUME"),
        "Last_Year_Period_Spends_old",
        "Last_Year_Period_GPS_old",
        "Last_Year_Period_NetSales_old",
        "Last_Year_Period_Volume_old",
        "Spend_change",
        "Vol_change",
        "NS_Change",
        "PRICE",
        sf.col("MC").alias("EC"),
        "EC_PER_QTYCASE",
        sf.col("MC_new").alias("MC"),
    )
    .distinct()
    .withColumn("MODELINGNAME", sf.lit(input_params["optimizationInput"]["spendtype"]))
    .withColumn("SCENARIO_NAME", sf.lit(scenario_name))
    .withColumn("DATE", sf.lit(input_params["optimizationInput"]["max"]))
)
# opti_output.display()


status_df = spark.createDataFrame(
    pd.DataFrame(
        {
            "SCENARIONAME": [scenario_name],
            "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
            "PROGRESSMESSAGE": [0.80],
        }
    )
)
# status_df.display()
data = (
    spark.read.format("snowflake")
    .options(**options)
    .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
    .load()
    .filter(sf.col("SCENARIONAME") != scenario_name)
)
sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
sf_output.write.format("snowflake").options(**options).option(
    "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
).mode("overwrite").save()


### Storing Table DF

# COMMAND ----------

spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt')).display()

# COMMAND ----------

# (opti_output.join(
#         decomp_df_0_0.select(
#             "BUSINESS",
#             "CATEGORY",
#             "SUBCATEGORY",
#             "SEGMENT",
#             "SHISHA",
#             "CLUSTER",
#             "CHANNEL",
#             "PLANNINGCUSTOMER",
#             "MODELINGNAME",
#             sf.col("VOLUME").alias("Decomp_Volume"),
#             sf.col("SPENDS").alias("Decomp_Spends"),
#             sf.col("GPS").alias("Decomp_GPS"),
#             sf.col("NETSALES").alias("Decomp_NetSales"),
#         ),
#         [
#             "BUSINESS",
#             "CATEGORY",
#             "SUBCATEGORY",
#             "SEGMENT",
#             "SHISHA",
#             "CLUSTER",
#             "CHANNEL",
#             "PLANNINGCUSTOMER",
#             "MODELINGNAME",
#         ],
#         "left",
#     )).withColumn("Spends_match", sf.when(sf.round(sf.col("Last_Year_Period_Spends_old"),0) == sf.round(sf.col("Decomp_Spends"),0), 1).otherwise(0)).withColumn("Volume_match", sf.when(sf.round(sf.col("Last_Year_Period_Volume_old"),0) == sf.round(sf.col("Decomp_Volume"),0), 1).otherwise(0)).withColumn("NetSales_match", sf.when(sf.round(sf.col("Last_Year_Period_NetSales_old"),0) == sf.round(sf.col("Decomp_NetSales"),0), 1).otherwise(0)).withColumn("GPS_match", sf.when(sf.round(sf.col("Last_Year_Period_GPS_old"),0) == sf.round(sf.col("Decomp_GPS"),0), 1).otherwise(0)).filter("Spends_match == 0 or Volume_match == 0 or NetSales_match == 0 or GPS_match == 0").display()

# COMMAND ----------

last_year_period

# COMMAND ----------

(spark.sql("select * from vw_decomp_df_20jun25").withColumn(
    "STARTDATE",
    sf.to_date(
        sf.concat_ws("/", sf.lit("01"), sf.col("MONTH"), sf.col("YEAR")), "dd/MM/yyyy"
    ),
).filter(
    (sf.col("STARTDATE").isin(last_year_period))
    & (sf.col("MODELINGNAME").isin(spend_list))
    & (sf.col("BUSINESS") == business_type)
    & (sf.col("CATEGORY").isin(category_list))
    & (sf.col("CHANNEL").isin(channel_list))
    & (sf.col("SHISHA").isin(shisha_list))
    & (sf.col("SPENDS") > 0)
    & (sf.col("VOLUME") >= 0)
).groupBy(
        "BUSINESS",
        "CATEGORY",
        "SUBCATEGORY",
        "SEGMENT",
        "SHISHA",
        "CLUSTER",
        "CHANNEL",
        "PLANNINGCUSTOMER",
        "MODELINGNAME",
    )
    .agg(
        sf.sum("SPENDS").alias("SPENDS"),
        sf.sum("GPS").alias("GPS"),
        sf.sum("NETSALES").alias("NETSALES"),
        sf.sum("VOLUME").alias("VOLUME"),
        sf.avg("PRICE").alias("PRICE"),
        sf.sum("EC").alias("EC"),
        sf.sum("EC_PER_QTYCASE").alias("EC_PER_QTYCASE"),
        sf.sum("MC").alias("MC"),
    )).filter("SEGMENT == 'JP/C&B/RTD/LargePET/PPP PET' and PLANNINGCUSTOMER == 'L5 JP NS Marui Honbu'").display()

# COMMAND ----------

iter2_opti_output = spark.sql("select * from {0}.{1}".format(opt_database_name, second_iter_data))
out_data.columns

# COMMAND ----------

# DBTITLE 1,final output
iter2_opti_output = (spark.sql
    ("select * from {0}.{1}".format(opt_database_name, second_iter_data))
    .drop("Spend_Change", "Vol_Change", "NS_Change")
    .withColumn(
        "Spend_Change",
        (sf.col("Optimized_Spends") - sf.col("Last_Year_Period_Spends_old"))
        / sf.col("Last_Year_Period_Spends_old"),
    )
    .withColumn(
        "NetSales_Change",
        (sf.col("Optimized_NetSales") - sf.col("Last_Year_Period_NetSales_old"))
        / sf.abs(sf.col("Last_Year_Period_NetSales_old")),
    )
    .withColumn(
        "Optimized_Spends",
        sf.when(
            (sf.col("Spend_Change") >= 0) & (sf.col("NetSales_Change") < 0),
            sf.col("Last_Year_Period_Spends_old"),
        ).otherwise(sf.col("Optimized_Spends")),
    )
    .withColumn(
        "Optimized_Volume",
        sf.when(
            (sf.col("Spend_Change") >= 0) & (sf.col("NetSales_Change") < 0),
            sf.col("Last_Year_Period_Volume_old"),
        ).otherwise(sf.col("Optimized_Volume")),
    )
    .withColumn(
        "Optimized_GPS",
        sf.when(
            (sf.col("Spend_Change") >= 0) & (sf.col("NetSales_Change") < 0),
            sf.col("Last_Year_Period_GPS_old"),
        ).otherwise(sf.col("Optimized_GPS")),
    )
    .withColumn(
        "Optimized_NetSales",
        sf.when(
            (sf.col("Spend_Change") >= 0) & (sf.col("NetSales_Change") < 0),
            sf.col("Last_Year_Period_NetSales_old"),
        ).otherwise(sf.col("Optimized_NetSales")),
    )
)
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
    .filter(
        (sf.col("STARTDATE").isin(last_year_period))
        & ~(sf.col("MODELINGNAME").isin(spend_list))
        & (sf.col("BUSINESS") == business_type)
        & (sf.col("CATEGORY").isin(category_list))
        & (sf.col("CHANNEL").isin(channel_list))
        & (sf.col("SHISHA").isin(shisha_list))
        # & (sf.col("SPENDS") > 0)
        # & (sf.col("VOLUME") >= 0)
    )
)
baseline = decomp_model_output.groupBy("SHISHA", "PLANNINGCUSTOMER", "SEGMENT").agg(
    sf.sum("Volume").alias("Baseline_Volume")
)
opt_updt_final = (
    iter2_opti_output.fillna(
        0,
        subset=[
            "Last_Year_Period_Spends_old",
            "Last_Year_Period_Volume_old",
            "Last_Year_Period_GPS_old",
            "Last_Year_Period_NetSales_old",
            "Optimized_Spends",
            "Optimized_Volume",
            "Optimized_GPS",
            "Optimized_NetSales",
        ],
    )
    .withColumn(
        "Spend_Change",
        (sf.col("Optimized_Spends") - sf.col("Last_Year_Period_Spends_old"))
        / sf.col("Last_Year_Period_Spends_old"),
    )
    .withColumn(
        "Volume_Change",
        (sf.col("Optimized_Volume") - sf.col("Last_Year_Period_Volume_old"))
        / sf.col("Last_Year_Period_Volume_old"),
    )
    .withColumn(
        "GPS_Change",
        (sf.col("Optimized_GPS") - sf.col("Last_Year_Period_GPS_old"))
        / sf.col("Last_Year_Period_GPS_old"),
    )
    .withColumn(
        "NetSales_Change",
        (sf.col("Optimized_NetSales") - sf.col("Last_Year_Period_NetSales_old"))
        / sf.abs(sf.col("Last_Year_Period_NetSales_old")),
    )
    .join(baseline, ["SHISHA", "PLANNINGCUSTOMER", "SEGMENT"], "left")
    .fillna(
        0,
        subset=[
            "Baseline_Volume",
            "Last_Year_Period_Spends_old",
            "Last_Year_Period_Volume_old",
            "Last_Year_Period_GPS_old",
            "Last_Year_Period_NetSales_old",
            "Optimized_Spends",
            "Optimized_Volume",
            "Optimized_GPS",
            "Optimized_NetSales",
            "Spend_Change",
            "Volume_Change",
            "GPS_Change",
            "NetSales_Change",
        ],
    )
    .withColumn("Total_initial_spend", sf.col("Last_Year_Period_Spends_old"))
    .withColumn(
        "Total_initial_volume",
        sf.col("Last_Year_Period_Volume_old") + sf.col("Baseline_Volume"),
    )
    .withColumn(
        "Total_initial_GPS",
        sf.col("Total_initial_volume")
        * sf.col("Last_Year_Period_GPS_old")
        / sf.col("Last_Year_Period_Volume_old"),
    )
    .fillna(0, subset=["Total_initial_volume", "Total_initial_GPS", "Total_initial_spend"])
    .withColumn(
        "Total_initial_netsales",
        sf.col("Total_initial_GPS") - sf.col("Total_initial_spend"),
    )
    .withColumn("Total_optimized_spend", sf.col("Optimized_Spends"))
    .withColumn(
        "Total_optimized_volume", sf.col("Optimized_Volume") + sf.col("Baseline_Volume")
    )
    .withColumn(
        "Total_optimized_GPS",
        sf.col("Total_optimized_volume")
        * sf.col("Last_Year_Period_GPS_old")
        / sf.col("Last_Year_Period_Volume_old"),
    )
    .fillna(0, subset=["Total_optimized_volume", "Total_optimized_GPS", "Total_optimized_spend"])
    .withColumn(
        "Total_optimized_NetSales",
        sf.col("Total_optimized_GPS") - sf.col("Total_optimized_spend"),
    )
    .withColumn(
        "Final_Spend_change",
        sf.col("Total_optimized_spend") / sf.col("Total_initial_spend") - sf.lit(1),
    )
    .withColumn(
        "Final_Volume_change",
        sf.col("Total_optimized_volume") / sf.col("Total_initial_volume") - sf.lit(1),
    )
    .withColumn(
        "Final_GPS_change",
        sf.col("Total_optimized_GPS") / sf.col("Total_initial_GPS") - sf.lit(1),
    )
    .withColumn(
        "Final_NetSales_change",
        (sf.col("Total_optimized_NetSales") - sf.col("Total_initial_NetSales"))
        / sf.abs(sf.col("Total_initial_NetSales")),
    )
    .fillna(
        0,
        subset=[
            "Final_Spend_change",
            "Final_Volume_change",
            "Final_GPS_change",
            "Final_NetSales_change",
        ],
    )
)
opt_updt_final.display()

table_name = opt_database_name + "." + second_iter_data + "_updt_final"
print(table_name)
opt_updt_final.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
    table_name
)

# COMMAND ----------

# opt_updt_final = (
#     opti_output.join(
#         decomp_df_0_0.select(
#             "BUSINESS",
#             "CATEGORY",
#             "SUBCATEGORY",
#             "SEGMENT",
#             "SHISHA",
#             "CLUSTER",
#             "CHANNEL",
#             "PLANNINGCUSTOMER",
#             "MODELINGNAME",
#             sf.col("VOLUME").alias("Decomp_Volume"),
#             sf.col("SPENDS").alias("Decomp_Spends"),
#             sf.col("GPS").alias("Decomp_GPS"),
#             sf.col("NETSALES").alias("Decomp_NetSales"),
#         ),
#         [
#             "BUSINESS",
#             "CATEGORY",
#             "SUBCATEGORY",
#             "SEGMENT",
#             "SHISHA",
#             "CLUSTER",
#             "CHANNEL",
#             "PLANNINGCUSTOMER",
#             "MODELINGNAME",
#         ],
#         "left",
#     )
#     .withColumn(
#         "Spends_adjustment_factor",
#         sf.col("Decomp_Spends")/sf.col("Last_Year_Period_Spends_old"),
#     )
#     .withColumn(
#         "Volume_adjustment_factor",
#         sf.col("Decomp_Volume")/sf.col("Last_Year_Period_Volume_old"),
#     )
#     .withColumn(
#         "GPS_adjustment_factor",
#         sf.col("Decomp_GPS")/sf.col("Last_Year_Period_GPS_old"),
#     )
#     .withColumn(
#         "NetSales_adjustment_factor",
#         sf.col("Decomp_NetSales")/sf.col("Last_Year_Period_Volume_old"),
#     )
#     .withColumn(
#         "Last_Year_Period_Spends_old_calc",
#         sf.col("Last_Year_Period_Spends_old")*sf.col("Spends_adjustment_factor"),
#     )
#     .withColumn(
#         "SPENDS",
#         sf.col("SPENDS")*sf.col("Spends_adjustment_factor"),
#     )
#     .withColumn(
#         "Last_Year_Period_GPS_old_calc", sf.col("Last_Year_Period_GPS_old")*sf.col("GPS_adjustment_factor")
#     )
#     .withColumn(
#         "GPS", sf.col("GPS")*sf.col("GPS_adjustment_factor")
#     )
#     .withColumn(
#         "Last_Year_Period_Volume_old_calc",
#         sf.col("Last_Year_Period_Volume_old")*sf.col("Volume_adjustment_factor"),
#     )
#     .withColumn(
#         "VOLUME",
#         sf.col("VOLUME")*sf.col("Volume_adjustment_factor"),
#     )
#     .withColumn(
#         "Decomp_NetSales_calc",
#         sf.col("Decomp_GPS") - sf.col("Decomp_Spends")
#     )
#     .withColumn(
#         "optimised_NetSales_adjustment_factor",
#         sf.col("Decomp_NetSales")/sf.col("Decomp_NetSales_calc"),
#     )
#     .withColumn(
#         "NETSALES",
#         sf.col("GPS")-sf.col("SPENDS")
#     )
#     .withColumn(
#         "NETSALES",
#         sf.col("NETSALES")*sf.col("optimised_NetSales_adjustment_factor")
#     )
#     .withColumn(
#         "Last_Year_Period_NetSales_old_calc",
#         sf.col("Last_Year_Period_NetSales_old")*sf.col("NetSales_adjustment_factor")
#     )
# )
# opt_updt_final.display()

# table_name = opt_database_name+'.'+second_iter_data+'_updt_final'
# print(table_name)
# opt_updt_final.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name)

# COMMAND ----------

spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt_final')).filter("PLANNINGCUSTOMER == 'L5 JP NS Matsumoto Honbu'").display()

# COMMAND ----------


# try:
    
#     opti_output = (spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt'))
#                 #    .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
#                    .select('BUSINESS','CATEGORY','SUBCATEGORY','SEGMENT','SHISHA','CLUSTER','CHANNEL','PLANNINGCUSTOMER',sf.col('Optimized_Spends').alias('SPENDS'),sf.col('Optimized_GPS_Adj').alias('GPS'),sf.col('Optimized_NetSales_Adj').alias('NETSALES'),sf.col('Optimized_Volume').alias('VOLUME'), 'Last_year_period_spends', 'Last_year_period_gps', 'Last_year_period_netsales', 'Last_year_period_volume', 'Spend_change', 'Volumen_change', 'GPS_chage', 'PRICE',sf.col('MC').alias('EC'),'EC_PER_QTYCASE',sf.col('MC_new').alias('MC'))
#                    .distinct()
#                    .withColumn('MODELINGNAME',sf.lit(input_params['optimizationInput']['spendtype']))
#                    .withColumn('SCENARIO_NAME',sf.lit(scenario_name))
#                    .withColumn('DATE', sf.lit(input_params['optimizationInput']['max']))
#                    )
#     opti_output.display()
    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.80]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.80]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()


# ### Storing Table DF



# COMMAND ----------

# opti_output = (spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt')))
# opti_output.display()

# COMMAND ----------


# try:
    
#     decomp_df_0_0 = decomp_df_base.withColumn(
#         "STARTDATE",
#         sf.to_date(
#             sf.concat_ws("/", sf.lit("01"), sf.col("MONTH"), sf.col("YEAR")), "dd/MM/yyyy"
#         ),
#     ).filter(
#         (sf.col("STARTDATE").isin(last_year_period))
#         & (sf.col("MODELINGNAME").isin(spend_list))
#         & (sf.col("BUSINESS") == business_type)
#         & (sf.col("CATEGORY").isin(category_list))
#         & (sf.col("CHANNEL").isin(channel_list))
#         & (sf.col("SHISHA").isin(shisha_list))
#         & (sf.col("SPENDS") > 0)
#         & (sf.col("VOLUME") >= 0)
#     )
#     decomp_df_0_0.display()

#     decomp_df_0_0 = decomp_df_0_0.groupBy("BUSINESS","CATEGORY","SUBCATEGORY","SEGMENT","SHISHA","CLUSTER","CHANNEL","PLANNINGCUSTOMER","MODELINGNAME").agg(sf.sum("SPENDS").alias("SPENDS"),sf.sum("GPS").alias("GPS"),sf.sum("NETSALES").alias("NETSALES"),sf.sum("VOLUME").alias("VOLUME"),sf.avg("PRICE").alias("PRICE"),sf.sum("EC").alias("EC"),sf.sum("EC_PER_QTYCASE").alias("EC_PER_QTYCASE"),sf.sum("MC").alias("MC")).withColumn('SCENARIO_NAME',sf.lit(scenario_name)).withColumn('DATE', sf.lit(input_params['optimizationInput']['max']))
#     decomp_df_0_0.display()

    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.82]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.82]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------



# try:
    
#     # df_add = decomp_df_0_0[~decomp_df_0_0.isin(opti_output).all(axis=1)]
#     # df_add.display()

#     # df_all = decomp_df_0_0.join(opti_output.drop_duplicates(), on=["BUSINESS","CATEGORY","SUBCATEGORY","SEGMENT","SHISHA","CHANNEL","PLANNINGCUSTOMER"], 
#     #                    how='left')
#     # opti_output.display()  

#     df_to_add = decomp_df_0_0.join(opti_output, on=["BUSINESS","CATEGORY","SUBCATEGORY","SEGMENT","SHISHA","CLUSTER","CHANNEL","PLANNINGCUSTOMER","MODELINGNAME"],how="left_anti")
#     df_to_add.display()

    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.84]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.84]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------



# try:
    
    
#     # opt_output_final = opti_data.unionByName(df_to_add, allowMissingColumns=True).distinct()
#     opt_output_final = opti_output.unionByName(df_to_add)
#     opt_output_final.display()

    
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [0.86]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
# except:
#     status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [0.86]}))
#     # status_df.display()
#     data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
#     sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
#     sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



# COMMAND ----------





# %md
### Storing Results in Back-end Tables

# COMMAND ----------

# print(opt_output_final.count())
# opt_output_final_1 = opt_output_final.join(decomp_df_0_0[['combined']].drop_duplicates().withColumn('check',sf.lit('1')), on='combined', how='left')
# opt_output_final_1 = opt_output_final_1[opt_output_final_1['check']=='1']
# opt_output_final_1 = opt_output_final_1.drop('combined')
# opt_output_final_1 = opt_output_final_1.drop('check')
# print(opt_output_final_1.count())
# # opt_output_final_1.display()

# COMMAND ----------

# # opt_output_final.display()
# # decomp_df_0_0.display()
# # list_a = decomp_df_0_0["combined"].to_list()
# a = list(decomp_df_0_0.select("combined"))

# # opt_output_final = opt_output_final.withColumn('combined', concat('SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'))
# # decomp_df_0_0 = decomp_df_0_0.withColumn('combined', concat('SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'))
# # a = decomp_df_0_0.select('combined').rdd.flatMap(lambda x: x).collect()
# print(opt_output_final.count())
# # opt_output_final_1 = opt_output_final[opt_output_final['combined'].isin(a)]
# opt_output_final_1 = opt_output_final.join(decomp_df_0_0[['combined']].drop_duplicates().withColumn('check',sf.lit('1')), on='combined', how='left')
# opt_output_final_1 = opt_output_final_1[opt_output_final_1['check']=='1']
# opt_output_final_1 = opt_output_final_1.drop('combined')
# opt_output_final_1 = opt_output_final_1.drop('check')
# print(opt_output_final_1.count())
# opt_output_final_1.display()
# opt_output_final_1[["check"]].drop_duplicates().display()
# # decomp_df_0_0[['combined']].drop_duplicates().withColumn('check',sf.lit('1')).display()

# COMMAND ----------

# # # a = decomp_df_0_0.select("combined").distinct()
# a = list(decomp_df_0_0['combined'])
# # a.display()
# # print(opt_output_final.count())
# # df1 = opt_output_final[~opt_output_final.combined.isin(decomp_df_0_0.select('combined').distinct().rdd.flatMap(lambda x: x).collect())]
# # # df1= opt_output_final.loc[opt_output_final["combined"].isin(decomp_df_0_0["combined"].unique())]
# # # opt_output_final_1 = opt_output_final[opt_output_final['combined'].isin(decomp_df_0_0['combined'])]
# # print(df1.count())

# COMMAND ----------

opti_data = spark.read.format("snowflake").options(**options).option("dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT").load().filter(sf.col('SCENARIO_NAME')!=scenario_name)
opt_updt_final.display()
opti_data.display()

# COMMAND ----------

# DBTITLE 1,optimization output write
months_list = [datetime.strptime(d, "%Y-%m-%d").month for d in last_year_period]
year_list = list(set([datetime.strptime(d, "%Y-%m-%d").year + 1 for d in last_year_period]))

# Get the last day of the month
last_day = pd.to_datetime(f"{int(year_) + 1}-{max_month}-01") + pd.offsets.MonthEnd(0)

# Add the last timestamp of that day (23:59:59.999999999)
end_of_period = pd.Timestamp(last_day.date()) + pd.Timedelta("23:59:59")

print(end_of_period)

try:

    # opt_output_final = opt_output_final.withColumn('combined', concat('SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'))
    # decomp_df_0_0 = decomp_df_0_0.withColumn('combined', concat('SEGMENT', 'SHISHA', 'PLANNINGCUSTOMER'))

    # print(opt_output_final.count())
    # opt_output_final_1 = opt_output_final.join(decomp_df_0_0[['combined']].drop_duplicates().withColumn('check',sf.lit('1')), on='combined', how='left')
    # opt_output_final_1 = opt_output_final_1[opt_output_final_1['check']=='1']
    # opt_output_final_1 = opt_output_final_1.drop('combined')
    # opt_output_final_1 = opt_output_final_1.drop('check')
    # print(opt_output_final_1.count())
    # opt_output_final_1.display()

    # a = decomp_df_0_0.select('combined').rdd.flatMap(lambda x: x).collect()
    # opt_output_final = opt_output_final[opt_output_final['combined'].isin(a)]
    # opt_output_final = opt_output_final.drop('combined')

    # opti_output = (spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt'))
    #             #    .withColumn('Optimized_GPS_Adj',sf.col('Optimized_GPS')*sf.col('Adjustment_Factor'))
    #                .select('BUSINESS','CATEGORY','SUBCATEGORY','SEGMENT','SHISHA','CLUSTER','CHANNEL','PLANNINGCUSTOMER',sf.col('Optimized_Spends').alias('SPENDS'),sf.col('Optimized_GPS_Adj').alias('GPS'),sf.col('Optimized_NetSales_Adj').alias('NETSALES'),sf.col('Optimized_Volume').alias('VOLUME'),'PRICE',sf.col('MC').alias('EC'),'EC_PER_QTYCASE',sf.col('MC_new').alias('MC'))
    #                .distinct()
    #                .withColumn('MODELINGNAME',sf.lit(input_params['optimizationInput']['spendtype']))
    #                .withColumn('SCENARIO_NAME',sf.lit(scenario_name))
    #                .withColumn('DATE', sf.lit(input_params['optimizationInput']['max']))
    #
    #           )
    table_name = opt_database_name + "." + second_iter_data + "_updt_final"
    print(table_name)
    opt_updt_final = spark.sql("select * from {}".format(table_name))
    opt_updt_final = (
        opt_updt_final.withColumn("DATE", sf.lit(end_of_period))
        .withColumn("YEAR", sf.year(sf.col("DATE")))
        .withColumn("MONTH", sf.month(sf.col("DATE")))
        .withColumn("HALF_YEAR", sf.when(sf.col("MONTH") <= 6, "H1").otherwise("H2"))
        .withColumn("SCENARIO_NAME", sf.lit(scenario_name))
    )
    # print(opt_updt_final.columns)
    opti_data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT")
        .load()
        .filter(sf.col("SCENARIO_NAME") != scenario_name)
        .withColumn(
            "IS_LATEST",
            sf.when(
                (
                    (sf.col("CATEGORY").isin(category_list))
                    & ((sf.col("BUSINESS").isin(business_type)))
                    & (sf.col("MODELINGNAME").isin(spend_list))
                    & (sf.col("SHISHA").isin(shisha_list))
                    & (sf.col("CHANNEL").isin(channel_list))
                    & (sf.month(sf.col("DATE")).isin(months_list))
                    & (sf.year(sf.col("DATE")).isin(year_list))
                ),
                sf.lit(0),
            ).otherwise(sf.col("IS_LATEST")),
        )
    )
    # opti_data.select("SCENARIO_NAME").distinct().display()
    # print(opti_data.columns)
    opt_output = opti_data.unionByName(
        opt_updt_final.withColumn(
            "MODELINGNAME", sf.concat_ws(",", opt_updt_final["MODELINGNAME"])
        ).withColumn("IS_LATEST", sf.lit(1)),
        allowMissingColumns=True,
    ).distinct()
    opt_output.filter(sf.col("SCENARIO_NAME") == scenario_name).display()
    opt_output.write.format("snowflake").options(**options).option(
        "dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT"
    ).mode("overwrite").save()
    print("Data loaded in backend table")

    # spark.read.format("snowflake").options(**options).option("dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT").load().filter(sf.col('SCENARIO_NAME')==scenario_name).display()

    # count_ini = spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt')).distinct().count()
    count_ini = (
        spark.sql(
            "select * from {0}.{1}".format(
                opt_database_name, second_iter_data + "_updt_final"
            )
        )
        .distinct()
        .count()
    )
    count_load = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "MID_JP_FF_OPTIMIZATION_OUTPUT")
        .load()
        .filter(sf.col("SCENARIO_NAME") == scenario_name)
        .count()
    )

    if count_ini == count_load:
        print("Data loaded successfully")

    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [0],
                "PROGRESSMESSAGE": [0.90],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()
except:
    status_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "SCENARIONAME": [scenario_name],
                "ISOPTIMIZATIONEXECUTIONCOMPLETED": [2],
                "PROGRESSMESSAGE": [0.90],
            }
        )
    )
    # status_df.display()
    data = (
        spark.read.format("snowflake")
        .options(**options)
        .option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER")
        .load()
        .filter(sf.col("SCENARIONAME") != scenario_name)
    )
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option(
        "dbtable", "STRATEGYACKNOWLEDGEMENTMASTER"
    ).mode("overwrite").save()

# COMMAND ----------

try:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [0],'PROGRESSMESSAGE': [1.0]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()
except:
    status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [2],'PROGRESSMESSAGE': [1.0]}))
    # status_df.display()
    data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
    sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
    sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()



status_df = spark.createDataFrame(pd.DataFrame({'SCENARIONAME': [scenario_name], 'ISOPTIMIZATIONEXECUTIONCOMPLETED': [1]}))
# status_df.display()
data = spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')!=scenario_name)
sf_output = data.unionByName(status_df, allowMissingColumns=True).distinct()
sf_output.write.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").mode("overwrite").save()

spark.read.format("snowflake").options(**options).option("dbtable", "STRATEGYACKNOWLEDGEMENTMASTER").load().filter(sf.col('SCENARIONAME')==scenario_name).show()






# COMMAND ----------

# out_data.display()
# decomp_df_base.display()
# decomp_df_base.filter(sf.col('YEAR')>=2024).display()

# COMMAND ----------

# decomp_output = (
#     spark.read.format("snowflake")
#     .options(**options)
#     .option("dbtable", "VW_DECOMP_OUTPUT")
#     .load()
# )
# decomp_output.display()

# COMMAND ----------

decomp_output = (spark.sql("select * from vw_decomp_df_20jun25"))
decomp_output.display()

# COMMAND ----------

decomp_output.filter("YEAR == '2024' and MONTH >= 7").groupBy("SEGMENT", "PLANNINGCUSTOMER", "SHISHA", "MODELINGNAME").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME")).display()

# COMMAND ----------

years = ['2022', '2023', '2024', '2025']
for year in years:
    print("Year = ", year)
    decomp_output.filter(sf.col("YEAR") == year).display()

# COMMAND ----------

opt_database_name+"."+second_iter_data+'_updt_final'

# COMMAND ----------

spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt_final')).display()

# COMMAND ----------

spark.sql("select * from {0}.mbt_wo_competition_new".format("tpo_output_v21")).display()

# COMMAND ----------

# spark.sql("select * from {0}.{1}".format(opt_database_name,second_iter_data+'_updt_final')).select("Date").distinct().display()

# COMMAND ----------

((spark.sql("select * from tpo_output_v21.mbt_wo_competition_price_corrected")).filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").withColumn('SPENDS',sf.col('PRICEOFF')+sf.col('PROMOTIONAD')+sf.col('LIQUIDATION'))
               .groupBy('YEAR','MONTH','SEGMENT','SHISHANM','PLANNINGCUSTOMER')
               .agg(sf.sum('SPENDS').alias('TOTAL_SPENDS'),
                    sf.sum('SALESAMT').alias('TOTAL_SALESAMT'),
                    sf.sum('SALESQTYCASE').alias('TOTAL_SALESQTYCASE'))
               .withColumn('YEAR',sf.col('YEAR').cast('integer'))
               .withColumn('MONTH',sf.col('MONTH').cast('integer'))
               .withColumnRenamed('SHISHANM','SHISHA')
               .distinct()).display()

# COMMAND ----------

(spark.sql("select * from tpo_output_v21.mbt_wo_competition_price_corrected")).filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").groupBy("SEGMENT", "PLANNINGCUSTOMER", "SHISHANM").agg(sf.sum("LIQUIDATION").alias("LIQUIDATION"), sf.sum("PROMOTIONAD").alias("PROMOTIONAD"), sf.sum("PRICEOFF").alias("PRICEOFF"), sf.sum("SALESAMT").alias("SALESAMT")).display()

# COMMAND ----------

spark.sql("select * from {0}.optimization_input".format(opt_database_name)).filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").groupBy("SEGMENT", "PLANNINGCUSTOMER", "SHISHA", "MODELINGNAME").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME")).display()

# COMMAND ----------

decomp_database_name = "tpo_output_v18"
decomp_df = spark.sql("select * from {0}.{1}".format(opt_database_name,decomp_table_name))
coef_info_df = spark.sql("select * from {0}.{1}".format(opt_database_name,data_prep_iter1))

decomp_df.filter("PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET' and YEAR == '2024' and MONTH >= 7").groupBy("SEGMENT", "PLANNINGCUSTOMER", "SHISHA", "MODELINGNAME").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES"), sf.sum("VOLUME").alias("VOLUME")).display()


# COMMAND ----------

(spark.sql("select * from {0}.output_final_with_ns".format("tpo_output_v21"))
               .filter("MODELINGNAME not in ('Base_BasePrice','Volume','PREDICTED_SALES')")
               .distinct()).filter("YEAR == '2024' and MONTH >= 7").groupBy("SEGMENT", "PLANNINGCUSTOMER", "SHISHA", "MODELINGNAME").agg(sf.sum("SPENDS").alias("SPENDS"), sf.sum("GPS").alias("GPS"), sf.sum("NETSALES").alias("NETSALES")).display()

# COMMAND ----------

# PLANNINGCUSTOMER == 'L5 JP NS Beisia Honbu Honten' and SEGMENT == 'JP/C&B/RTD/LargePET/Mainstream PET'
