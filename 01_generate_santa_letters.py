# Databricks notebook source
# MAGIC %pip install databricks-genai-inference==0.1.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Generate Santa Letters using Foundational Models API (Llama2-70B-Chat). 
# MAGIC 1. Generate dataset of popular names
# MAGIC 2. Generate dataset of Christmas gift topics
# MAGIC 3. Combine both datasets 

# COMMAND ----------

# MAGIC %md ## Generate popular kid names

# COMMAND ----------

import mlflow.gateway
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

from databricks_genai_inference import ChatCompletion
import os

os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

named_prompt = """What are the 100 most popular children's first name in North America over the past 20 years?"""
response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are a helpful assistant."},
                                           {"role": "user","content": named_prompt}],
                                 max_tokens=1500).message
print(f"response.message:{response}")

# COMMAND ----------

import re

result_str = ""

temp = re.findall('\d+\.\s*(.*)', response)

for item in temp:
    result_str += item + ", "

result_str = result_str[:-2]  # removing the extra comma and space at the end
final_names = [result.replace(" ","") for result in result_str.split(",")]
print(f"Final names: {final_names}")

# COMMAND ----------

# MAGIC %md ## Generate gift themes for Christmas

# COMMAND ----------

gift_prompt = """What are the 20 most popular children's gift topics between the ages 5-15, in North America in 2023? Do not output gift topics such as Home decor, organization, Travel accessories, Subscription boxes, or Personalized items in the response."""

gift_response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are a helpful assistant."},
                                           {"role": "user","content": gift_prompt}],
                                 max_tokens=1500,
                                 temperature=0.7).message

# print(f"response.message:{gift_response}")

gift_ideas = re.findall('\d+\.\s*(.*)', gift_response)
print(gift_ideas)

# COMMAND ----------

df2 = spark.createDataFrame([(x,) for x in gift_ideas], ["gift_topics"])
df2.write.format("delta").mode("overwrite").saveAsTable("ai_blog.gen_data.gift_topics2")

# COMMAND ----------

# MAGIC %md ## Create prompt that inputs the child's name and gift theme into LLM

# COMMAND ----------

system_prompt = """You are an AI assistant, helping children write letters to Santa asking for Christmas Presents
  Use the kid_name as the child's name provided in the instructions below
  Use the gift_theme provided as the Christmas present category
  Use language that a child would and do not exhibit sexist, racist, violent or any offensive langauge in these letters. Keep it at a length, a child would."""

def generate_letters(kid_name, gift_theme):

    def fill_prompt(kid_name: str, gift_theme: str) -> str:
        template = f"""Child's name: {kid_name} Christmas present category: {gift_theme} [/INST]
        """
        return template

    letter_prompt = fill_prompt(kid_name, gift_theme)
    response = ChatCompletion.create(model="llama-2-70b-chat",
                                    messages=[{"role": "system", "content": system_prompt},
                                              {"role": "user","content": letter_prompt}],
                                    max_tokens=1500,
                                    temperature=0.8).message

    return response

# COMMAND ----------

generate_letters(kid_name="Alex", gift_theme="Sports equipment")

# COMMAND ----------

import random
import json
import pandas as pd

gift_topics_list = spark.table("ai_blog.gen_data.gift_topics3").toPandas()['gift_list'].to_list()
final_names_list = spark.table('ai_blog.gen_data.names').toPandas()['names'].to_list()

names_list = []
gift_list = []

for i in range(1000):
  names_list.append(random.choice(final_names_list))
  gift_list.append(random.choice(gift_topics_list))
# for name in final_names_list:
#   names_list.append(name)
#   gift_list.append(random.choice(gift_topics_list))

pandas_df = pd.DataFrame({'name': names_list,
                          'gift_list': gift_list})

# spark_df = spark.createDataFrame(pandas_df)
# display(spark_df)
pandas_df.head()

# COMMAND ----------

# MAGIC %md ### Using Spark UDF to distribute calls, this only works on smaller datasets as timeout errors can typically occur

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StringType

def generate_letters(kid_name, gift_theme):
    
    def fill_prompt(kid_name: str, gift_theme: str) -> str:
        template = f"""[INST] <<SYS>>
        You are an AI assistant, helping children write letters to Santa asking for Christmas Presents
        Use the kid_name as the child's name provided in the instructions below
        Use the gift_theme provided as the Christmas present category
        Use language that a child would and do not exhibit sexist, racist, violent or any offensive langauge in these letters. Keep it at a length, a child would.
        <</SYS>>
        Child's name: {kid_name} Christmas present category: {gift_theme} [/INST]
        """
        return template

    letter_prompt = fill_prompt(kid_name, gift_theme)
    response = mlflow.gateway.query(route="mosaicml-llama2-70b-completions", data={"prompt": letter_prompt, "temperature": 0.8, "max_tokens": 1500})['candidates'][0]['text']
    return response

generate_letters_udf = F.udf(generate_letters, StringType())

# spark_df = spark.createDataFrame(pandas_df)
# sample_df = spark_df.limit(20)
letter_df = spark_df \
  .cache()

letter_df = letter_df \
  .withColumn('letters', generate_letters_udf(F.col('name'), F.col('gift_list')))
  # .cache()

letter_df.write.format('delta').mode('overwrite').saveAsTable('ai_blog.gen_data.santa_letters')
# print(letter_df.count())
letter_df = spark.table('ai_blog.gen_data.santa_letters')
display(letter_df)
