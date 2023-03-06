from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

import plotly.express as px
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import col
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
import pyspark.sql.types as T

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
        html.H2("Predicting fare and trip time with a Machine Learning Model (on Databricks)"),
        html.Span([
            "Distance [miles]: ",
            dcc.Input(id="trip-distance", value="1", type="number")
        ]),
        html.Br(),
        html.Span([
            "Estimated Time [min]: ",
            html.Span(id="estimated-time")
        ])
        ,
        html.Br(),
        html.Span([
            "Estimated Fare [dollars]: ",
            html.Span(id="estimated-fare")
        ])
    ]),

    html.Div([
        html.H2("NYC Taxi analysis (data processing on Databricks)"),
        dcc.Dropdown(['pickup_zip', 'dropoff_zip'], 'dropoff_zip', id='Map-dropdown-zip'),
        dcc.Dropdown(['avg_trip_duration','cout_trips', 'avg_trip_distance'], 'avg_trip_duration', id='Map-dropdown-agg'),
        dcc.Graph(id="NYC-zip-map-analysis")
    ]),

    html.Div([
    html.H2("Community contribution: Add your trip to the New York Taxi database (on Databricks)"),
    html.Div(["Pick up zip code: ", dcc.Input(id='input-on-submit-pickup_zip', type="text")]), 
    html.Div(["Drop off zip code: ", dcc.Input(id='input-on-submit-dropoff_zip', type="text")]), 
    html.Div(["Distance [miles]: ", dcc.Input(id='input-on-submit-trip_distance', type="text")]), 
    html.Div(["Trip duration [min]: ", dcc.Input(id='input-on-submit-trip_duration', type="text")]), 
    html.Div(["Fare [dollars]: ", dcc.Input(id='input-on-submit-fare_amount', type="text")]), 
    html.Div(["Update Databricks database by clicking: ",html.Button('Submit', id='submit-val', n_clicks=0)]),
    html.Div(id='container-button-basic', children='Enter values and press submit'),
    html.Div(["data saved in E2dogfood table: hive_metastore.da_vladislav_manticlugo_8874_asp.nyctaxi_data"])
    ]),
   

    html.Div([
        html.H2("Trips by pickup zip(data processing on Databricks)"),
        html.Span([
            "Count >= ",
            dcc.Input(id="count-input", value="10", type="number"),
        ]),
        dcc.Graph(id="postcode-trip-count")
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
    [Output("estimated-time", "children"),
     Output("estimated-fare", "children")],
    Input("trip-distance", "value")
)

def predict_time(distance):
    with open("cluster.json") as f:
        config = json.load(f)
        host = config["workspaceUrl"]
        token = config["token"]
    import os
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["DATABRICKS_HOST"] = f"https://{host}"
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    import mlflow
    import pandas as pd
    model_name = "NYTaxi_duration"
    X = pd.DataFrame([distance], columns = ['trip_distance'])
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
    pred1 = model.predict(X)
    model_name = "NYTaxi_fare_amount"
    X = pd.DataFrame([distance], columns = ['trip_distance'])
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
    pred2 = model.predict(X)

    return [pred1[0], pred2[0]]


@app.callback(
    Output("NYC-zip-map-analysis", "figure"),
    [Input('Map-dropdown-zip', 'value'),
     Input('Map-dropdown-agg', 'value')]
)

def update_output(zip_plot, column_map_show):
    df = spark.read.table("samples.nyctaxi.trips")
    df = df.withColumn('DurationInMin',F.round((F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))/60,2))
    df = df.where(df.DurationInMin<200)
    df = df.withColumn("day_of_week", F.date_format("tpep_pickup_datetime", 'E'))
    # zip_plot = 'pickup_zip'
    # zip_plot = 'dropoff_zip'
    df_map = (df.groupBy(col(zip_plot).alias('zip_code')).agg(
    F.avg('DurationInMin').alias('avg_trip_duration'),
    F.count('DurationInMin').alias('cout_trips'),
    F.avg('trip_distance').alias('avg_trip_distance'))
    )

    df_map_show = df_map.toPandas()
    df_map_show.head()

    nycmap = json.load(open("nyc-zip-code-tabulation-areas-polygons.geojson"))
    return px.choropleth_mapbox(df_map_show,
                            geojson=nycmap,
                            locations="zip_code",
                            featureidkey="properties.postalCode",
                            color=column_map_show,
                            color_continuous_scale="viridis",
                            mapbox_style="carto-positron",
                            zoom=9, center={"lat": 40.7, "lon": -73.9},
                            opacity=0.7,
                            hover_name="zip_code"
                            )

@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit-pickup_zip', 'value'),
    State('input-on-submit-dropoff_zip', 'value'),
    State('input-on-submit-trip_distance', 'value'),
    State('input-on-submit-trip_duration', 'value'),
    State('input-on-submit-fare_amount', 'value')
    
)
def update_output(n_clicks,pickup_zip,dropoff_zip,trip_distance,trip_duration,fare_amount):
    df = spark.read.table("hive_metastore.da_vladislav_manticlugo_8874_asp.nyctaxi_data")
    schema = T.StructType([ \
    T.StructField("trip_distance",T.DoubleType(),True), \
    T.StructField("fare_amount",T.DoubleType(),True), \
    T.StructField("pickup_zip",T.IntegerType(),True), \
    T.StructField("dropoff_zip", T.IntegerType(), True), \
    T.StructField("trip_duration", T.DoubleType(), True),\
    ])

    df_row_data = ([[
    float(trip_distance),
    float(fare_amount),
    int(pickup_zip),
    int(dropoff_zip),
    float(trip_duration)
    ]])
    df_row_data
    df2 = (spark.createDataFrame(data = df_row_data,schema = schema)
    .withColumn('tpep_pickup_datetime',F.current_timestamp())
    .withColumn('tpep_dropoff_datetime',F.to_timestamp(F.from_unixtime(F.unix_timestamp(F.current_timestamp())+trip_duration*60)))
    )
    df2 = df2.select(df.columns)
    df2 = df.union(df2)
    # df2.write.saveAsTable("hive_metastore.da_vladislav_manticlugo_8874_asp.nyctaxi_data", mode='overwrite', format='delta') 
    # #Db connect error on grpc needs to be solved before writing to tables

    return (f"The submitted value to database are: \n"
            f"pickup_zip = {pickup_zip},"  
            f"dropoff_zip = {dropoff_zip},"  
            f"dropoff_zip = {dropoff_zip},"  
            f"trip_distance = {trip_distance},"
            f"trip_duration = {trip_duration},"
            f"fare_amount = {fare_amount},"
            f"-> Submitted={n_clicks}")


if __name__ == "__main__":
    app.run_server(debug=True)
