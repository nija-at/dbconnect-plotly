# Databricks Connect x Plotly

Demonstration of using Databricks Connect V2 on Plotly.

## Prerequisites

You need to have the latest DB Connect Python tar file to run the application.
If you don't have this, contact the authors - https://databricks.atlassian.net/wiki/spaces/UN/pages/2977858055/

## Running the app

Setup and activate a Conda environment.

```shell
conda env create -f ./conda.yml
conda activate dbconnect-plotly
```

Install DB Connect v2

```shell
pip install <dbconnect python tar>
```

Run the plotly app.

```shell
python app.py
```
