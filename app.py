from dash import Dash, html, dcc
import plotly.express as px
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

def spark_session():
    host = "<databricks workspace url>"
    clusterId = "<cluster id>"
    pat = "<token>"

    connStr = f"sc://{host}:443/;token={pat};x-databricks-cluster-id={clusterId}"

    return SparkSession.builder.remote(connStr).getOrCreate()

def pickupzip_sample(spark: SparkSession):
    df = spark.read.table("samples.nyctaxi.trips")
    df = df.withColumn("pickup_zip", col("pickup_zip").cast(StringType()))
    return df.groupby("pickup_zip").count().limit(5)


spark = spark_session()
sample = pickupzip_sample(spark).toPandas()

app = Dash(__name__)

fig = px.bar(sample, x="pickup_zip", y="count", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
