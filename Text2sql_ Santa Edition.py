# Databricks notebook source
# %pip install databricks-vectorsearch-preview
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch==0.21 mlflow==2.8.0 databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datasets import load_dataset , Dataset, concatenate_datasets 
import pandas as pd
#import sqlite3

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC USE CATALOG ai_blog;
# MAGIC USE SCHEMA gen_data;
# MAGIC
# MAGIC -- -- CREATE A CATALOG, SCHEMA AND VOLUME TO STORE DATA NEEDED FOR THIS. IN PRACTICE, YOU COULD USE AN EXISTING VOLUME
# MAGIC -- CREATE CATALOG IF NOT EXISTS structured_rag;
# MAGIC -- USE CATALOG structured_rag;
# MAGIC -- CREATE DATABASE IF NOT EXISTS structured_rag_db;
# MAGIC -- USE DATABASE structured_rag_db;
# MAGIC -- CREATE VOLUME IF NOT EXISTS structured_rag_raw;

# COMMAND ----------

# conn = sqlite3.connect('/Volumes/structured_rag/structured_rag_db/structured_rag_raw/database.sqlite')

# COMMAND ----------

# # Query to fetch the table names from the SQLite master table.
# query = "SELECT name FROM sqlite_master WHERE type='table';"
# # Execute the query and fetch all table names.
# table_names = conn.execute(query).fetchall()
# table_names.pop(0)
# table_names

# COMMAND ----------

for table in table_names:
  table_name = table[0]
  
  # Query to fetch all data from the current table.
  query = f"SELECT * FROM {table_name};"
  
  # Use Pandas to read data from the SQLite database into a DataFrame.
  df = pd.read_sql_query(query, conn)
  spark.createDataFrame(df).write.saveAsTable(table_name)

# COMMAND ----------

# Use spark SQL to show all tables in the specified database.
tables = spark.sql(f"SHOW TABLES")

# Display the list of tables.
tables.show()

# COMMAND ----------

tables = spark.sql("SHOW TABLES IN ai_blog.gen_data")
table_names = [row.tableName for row in tables.collect()]
table_names = [item for item in table_names if item in ['gift_topics3', 'names', 'santa_letters_final_processed','top_santa_items']]
table_names

# COMMAND ----------

# MAGIC %md ### Extract column name and comments from tables

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

table_metadata = []

for (i, tbl) in enumerate(table_names):
  tbl_columns = spark.table(f"{tbl}").columns
  desc_table_ext = spark.sql(f"DESCRIBE TABLE EXTENDED {tbl}")
  tbl_desc = f"Table name: {tbl}"
  tbl_metadata = (desc_table_ext \
    .withColumn('id', F.lit(i))
    .withColumn("table_name", F.lit(tbl_desc)) \
    .crossJoin(desc_table_ext.filter(F.col("col_name")=="Comment").select(F.col("data_type").alias("table_comment")))
    .filter(F.col("col_name").isin(tbl_columns))
    .groupby("id", "table_name", "table_comment") 
    .agg(F.collect_list("col_name").alias("column_names"),
         F.collect_list("data_type").alias("data_types"),
         F.collect_list("comment").alias("comments"))
    .withColumn("column_names_str", F.concat_ws(";", "column_names"))
    .withColumn("data_types_str", F.concat_ws(";", "data_types"))
    .withColumn("comments_str", F.concat_ws(";", "comments"))
    .withColumn("table_description", F.concat(F.col("table_name"), F.lit("\n"),
                                              F.lit("Table comment:\n "), F.col("table_comment"), F.lit("\n"),
                                              F.lit("Column names:\n "), F.col("column_names_str"), F.lit("\n"),
                                              F.lit("Column descriptions:\n "), F.col("comments_str"), F.lit("\n"),
                                              F.lit("Column data types:\n "), F.col("data_types_str"), F.lit("\n"),
                                              )
                )
    )
  
  table_metadata.append(tbl_metadata)

metadata_df = reduce(DataFrame.union, table_metadata)
display(metadata_df)
# metadata_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("table_descriptions2")

# COMMAND ----------

display(spark.table("table_descriptions2"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE table_descriptions2 SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %sql describe table extended table_descriptons2

# COMMAND ----------

# MAGIC %md ### Older code

# COMMAND ----------

# # Get a list of table names in the schema
# tables = spark.sql("SHOW TABLES")
# table_names = [row.tableName for row in tables.collect()]
# table_names

# COMMAND ----------

# Initialize an empty list to store table descriptions
table_descriptions = []

# Loop over each table
for table_name in table_names:
    # Describe the table
    table_description = spark.sql(f"DESCRIBE {table_name}")
    
    # Create a sentence describing the table
    sentence = f"Table '{table_name}' has the following columns:"
    
    # Loop over columns in the table description
    for row in table_description.collect():
        # Extract column name and data type
        column_name, data_type, _ = row
        sentence += f"\n- Column '{column_name}' with data type '{data_type}'"
    
    # Add the table description sentence to the list
    table_descriptions.append(sentence)

# COMMAND ----------

#Verify if it worked or not
table_descriptions

# COMMAND ----------

ids = list(range(3))
len(ids)== len(table_descriptions)

# COMMAND ----------

index_dict = {'ids': ids, 'table_descriptions': table_descriptions}
spark.createDataFrame(pd.DataFrame.from_dict(index_dict)).write.saveAsTable('table_descriptions')

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM table_descriptions

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE table_descriptions SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md ### Create embedding and model-serving end-point (Vector Search is not setup on Azure yet!)

# COMMAND ----------

scope = "dbdemos"
key = "mosaic_ml_api_key"

# COMMAND ----------

#init MLflow experiment
import mlflow
from mlflow import gateway

gateway.set_gateway_uri(gateway_uri="databricks")
#define our embedding route name, this is the endpoint we'll call for our embeddings
mosaic_embeddings_route_name = "mosaicml-instructor-xl-embeddings_abm"

try:
    route = gateway.get_route(mosaic_embeddings_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_embeddings_route_name}")
    print(gateway.create_route(
        name=mosaic_embeddings_route_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "instructor-xl",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope=scope, key=key)#Don't have a MosaicML Key ? Try with AzureOpenAI instead!
            }
        }
    ))

# COMMAND ----------

# MAGIC %md ### Creating an embedding model serving endpoint between Vector Search Index and AI Gateway (doesnt work on Azure, need foundational models)

# COMMAND ----------

from databricks_genai_inference import Embedding
# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
# Databricks genai sdk makes it easy to create your embeddings:

# NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = Embedding.create(model="bge-large-en", input=["What is Apache Spark?"])
print(embeddings)

# COMMAND ----------

# MAGIC %md ### Work around: use mlflow AI gateway MosaicML embedding

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, FloatType

def get_embedding(description: str):
  response = gateway.query(route=mosaic_embeddings_route_name, data={"text": description})
  return response['embeddings']

get_embedding_udf = F.udf(get_embedding, ArrayType(FloatType()))
embedded_df = spark.table("table_descriptions2") \
  .withColumn("embeddings", get_embedding_udf("table_description")) \
  .select("id", "table_description", "embeddings")

display(embedded_df)

# COMMAND ----------

catalog = "ai_blog"
schema = "gen_data"
table_name = "embedded_table_descriptions"

embedded_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table_name}")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE embedded_table_descriptions SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md ### Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_endpoint_name = "text2sql"

vsc = VectorSearchClient()
# vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

if vs_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
# print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md ### Create self-managed vector search using endpoint (using pre-built embedding model enpoint)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


catalog = "ai_blog"
schema = "gen_data"
table_name = "embedded_table_descriptions"
vs_endpoint_name = "text2sql"

#The table we'd like to index
source_table_fullname = f"{catalog}.{schema}.{table_name}"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{schema}.{table_name}_index"

# vsc.create_delta_sync_index()
vsc.create_delta_sync_index(
  endpoint_name=vs_endpoint_name,
  index_name=vs_index_fullname,
  source_table_name=source_table_fullname,
  pipeline_type="CONTINUOUS",
  primary_key="id",
  embedding_source_column="table_description",
  embedding_vector_column="embeddings",
  embedding_dimension=768
)

# COMMAND ----------

import mlflow
from mlflow import gateway

question = "Write a SQL query to find number of letters Santa has from kids?"

def get_embedding(description: str):
  response = gateway.query(route=mosaic_embeddings_route_name, data={"text": description})
  return response['embeddings']

# e = Embedding.create(model="bge-large-en", input=question)
e = get_embedding(question)

results = vsc.get_index(vs_endpoint_name, vs_index_fullname).similarity_search(
  query_vector=e,
  columns=["table_description"],
  num_results=1)
results

# COMMAND ----------

# context = results['result']['data_array'][0][0] + results['result']['data_array'][1][0]
context = results['result']['data_array'][0][0]
context

# COMMAND ----------

# results = client.similarity_search(
#   index_name = "vs_catalog.vs_schema.structuredrag-table-index",
#   query_text = "Write a SQL query to  find the number of soccer players",
#   columns = ["ids", "table_descriptions"], # columns to return
#   num_results = 2)

# results['result']['data_array'][0]

# COMMAND ----------

# context = results['result']['data_array'][0][1]+ ' \n '+results['result']['data_array'][1][1]
# print(context)


# COMMAND ----------

def fill_prompt(context: str, question: str) -> str:
    template = f"""[INST] <<SYS>>
You are an AI data analyst, helping business users by generating SQL queries based on their questions asked in English. 

Some information about possibly relevant tables will be provided to you, but you may or may not need all the tables provided for the SQL query.

Ensure the SQL queries you generate are accurate to the best of your ability

Please only print out the SQL query, nothing else

<</SYS>>

{context}

{question} [/INST]
"""
    return template

# COMMAND ----------

# question  = 'What is the number of soccer players?'
question = "What is the number of basketballs requested from final_list column in Santa letters?"
question = "Write a SQL query to find number of letters Santa has from kids?"

# COMMAND ----------

prompt = fill_prompt(context, question)
prompt

# COMMAND ----------

import mlflow.gateway
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# Text-to-SQL example query
question = "Write a SQL query to find out number of letters written by Elijah and name the column 'Elijah_count'?"
prompt = fill_prompt(context, question)

query= mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0.0
        },
    )

processed_query = query['candidates'][0]['text'].split(';')[0].replace('\n',' ').replace("```", "")
print(f"SQL query: {processed_query}")
display(spark.sql(processed_query))

# COMMAND ----------

# Text-to-SQL example query
question = "Write a SQL query to find out number of letters written by Elijah and Gabriel name the columns 'kid_letters'?"
prompt = fill_prompt(context, question)

query= mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0.0
        },
    )

processed_query = query['candidates'][0]['text'].split(';')[0].replace('\n',' ').replace("```", "")
print(f"SQL query: {processed_query}")
display(spark.sql(processed_query))

# COMMAND ----------

# Text-to-SQL example query
question = "Write a SQL query to find out number of letters grouped by each kid Elijah and Gabriel and name the columns 'kid_letters'. Make sure to use GROUP BY"
prompt = fill_prompt(context, question)

query= mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0.0
        },
    )

processed_query = query['candidates'][0]['text'].split(';')[0].replace('\n',' ').replace("```", "")
print(f"SQL query: {processed_query}")
display(spark.sql(processed_query))

# COMMAND ----------

query['candidates'][0]['text'].split(';')[0].replace('\n',' ').replace("```", "")

# COMMAND ----------

display(spark.sql(query['candidates'][0]['text'].split(';')[0].replace('\n',' ').replace("```", "")))
