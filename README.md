# 202112-19-Back-to-Normality-Covid19-Timeseries-Forecasting

## Introduction

The goal of this project is to provide a vision for the near future development of the pandemic in NYC to reduce uncertainty. We achieve this goal by develop time series forecasting and generating models with multiple types of time series. The final product that we produce is an automatic system which can update data, models, and predictions on daily basis.

## Analysis

This folder contains the ARIMA analysis and other baseline model (including regression model) that is used while we were researching for this project.

## CDC-Data-Analysis-N-Transformation

This folder contains the data analysis and transformation we did on CDC data. Most of the code were reused in developing the scripts for the DAG.

## Case-By-State

This folder contains the transformed data from CDC covid-19 cases publication. It has the csv files for each state's covid cases history and vaccination rate history.

## D3

This folder contains the prototype for our D3 visualization.

![Alt text](./d3.png?raw=true "D3 Visualization")

## DAG

In the DAG folder, there are the dag management system and the scripts that the dag system will run. After the scheduler starts, the system will collect data and store the transformed data into the script folder. It will then use the transformed data to train the VAR model and the RNN model.

![Alt text](./airflow_graph.png?raw=true "Airflow System")

## Data

Data folder contains the data we collected from NYCHealth. The scripts in DAG use the same data source.

## EDA_1st_2nd_report

This folder contains the jupyter notebooks of the analysis for the first and second reports.

## Processed_Data

Saved for the processed data from NYCHealth data.



