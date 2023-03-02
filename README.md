# Databricks Connect x Plotly

Demonstration of using Databricks Connect V2 on Plotly.

## Prerequisites

You need to have the latest DB Connect Python tar file to run the application.
If you don't have this, contact the authors - https://databricks.atlassian.net/wiki/spaces/UN/pages/2977858055/

## Set up Cluster and credentials

Create a cluster that has the Spark Connect running.

Generate a PAT token from the Databricks workspace.

Create a `cluster.json` file using the following command with the placeholder values filled in.

```shell
cat << EOF > cluster.json
{
  "workspaceUrl": "<workspace url>",
  "clusterId": "<cluster id>",
  "token": "<pat token>"
}
EOF
```

## Run the app

Create and activate a Conda environment.

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
