from dash import Dash, html, dcc, Output, Input
import plotly.express as px
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import col
from pyspark.sql.types import StringType

import json

def spark_session():
    with open("cluster.json") as f:
        config = json.load(f)

    host = config["workspaceUrl"]
    clusterId = config["clusterId"]
    token = config["token"]

    connStr = f"sc://{host}:443/;token={token};x-databricks-cluster-id={clusterId}"

    return SparkSession.builder.remote(connStr).getOrCreate()


spark = spark_session()

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children="Dash x Databricks Demo"),

    html.Div([
        html.H2("Trips by pickup zip"),
        html.Span([
            "Count >= ",
            dcc.Input(id="count-input", value="10", type="number"),
        ]),
        dcc.Graph(id="postcode-trip-count")
    ]),

    html.Div([
        html.H2("Predicting fare and trip time"),
        html.Span([
            "Distance: ",
            dcc.Input(id="trip-distance", value="", type="number")
        ]),
        html.Br(),
        html.Span([
            "Estimated Time: ",
            html.Span(id="estimated-time")
        ])
    ])
])


@app.callback(
    Output("postcode-trip-count", "figure"),
    Input("count-input", "value")
)
def update_trip_count(greaterThan):
    df = spark.read.table("samples.nyctaxi.trips")

    df = df.withColumn("pickup_zip", col("pickup_zip").cast(StringType())).withColumn("dropoff_zip", col("dropoff_zip").cast(StringType()))
    df = df.groupBy("pickup_zip", "dropoff_zip").count()
    df = df.filter(col("count") >= int(greaterThan))

    return px.scatter(df.toPandas(), x="pickup_zip", y="dropoff_zip", size="count", height=1000, width=1000)


@app.callback(
    Output("estimated-time", "children"),
    Input("trip-distance", "value")
)
def predict_time(distance):
    return distance * 2


if __name__ == "__main__":
    app.run_server(debug=True)
