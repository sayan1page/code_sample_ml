import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
import lightgbm as lgb
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsforecast.models import AutoARIMA
from datetime import datetime
import numpy as np

# Load the modified data
df = pd.read_csv('C:\\Users\\Personal\\Downloads\\scr-dataset.csv')
df['Date'] = pd.date_range(start=datetime(2018, 1, 1), periods=len(df), freq='M')

# Extend to x = 50
future_x = np.arange(df['x'].max() + 0.1, 50.1, 0.1)
future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=len(future_x), freq='M')
future_df = pd.DataFrame({'x': future_x, 'Date': future_dates})
combined_df = pd.concat([df, future_df], ignore_index=True)

# Initialize Dash app
app = dash.Dash("Demo")

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'stat', 'value': 'stat'},
            {'label': 'DL', 'value': 'dl'},
            {'label': 'ML', 'value': 'ml'}
        ],
        value='stat'
    ),
    dcc.Graph(id='price-graph'),
    html.Div(id='model-metrics', style={'padding': '20px'})
])


@app.callback(
    [Output('price-graph', 'figure'),
     Output('model-metrics', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_graph(selected_model):
    filtered_df = combined_df[combined_df['x'] <= 44.9].copy()
    filtered_df['y'].fillna(0, inplace=True)

    # Rename columns as required
    filtered_df.rename(columns={"x": "ds", "y": "y"}, inplace=True)

    # Ensure 'ds' is in datetime format
    filtered_df['ds'] = pd.to_datetime(filtered_df['Date'])

    if selected_model == 'stat':
        filtered_df["unique_id"] = 1
        models = [AutoARIMA()]
        sf = StatsForecast(models=models, freq='M')
        sf.fit(filtered_df[['unique_id', 'ds', 'y']])

        future_dates = pd.date_range(start=filtered_df['ds'].max() + pd.DateOffset(months=1), periods=len(future_x),
                                     freq='M')
        predictions = sf.predict(h=len(future_x))

        mse_value = mean_squared_error(filtered_df['y'].iloc[-len(future_x):], predictions['AutoARIMA'][:len(future_x)])
        r2_value = r2_score(filtered_df['y'].iloc[-len(future_x):], predictions['AutoARIMA'][:len(future_x)])

        # Extract the predicted value at x = 50
        predicted_value_at_50 = predictions['AutoARIMA'].iloc[-1]  # Last value corresponds to x=50

        metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}, Predicted Value at x=50: {predicted_value_at_50:.2f}"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Values'))
        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions['AutoARIMA'], mode='lines+markers', name='Predicted Values',
                       line=dict(dash='dash')))

        return fig, metrics

    if selected_model == 'dl':
        filtered_df["unique_id"] = 1
        horizon = 12
        models = [NBEATS(input_size=2 * horizon, h=horizon, max_steps=10),
                  NHITS(input_size=2 * horizon, h=horizon, max_steps=10)]
        nf = NeuralForecast(models=models, freq='M')
        nf.fit(df=filtered_df[['unique_id', 'ds', 'y']])
        predictions = nf.predict().reset_index()
        res = (predictions['NHITS'] + predictions['NBEATS']) / 2

        mse_value = mean_squared_error(filtered_df['y'].iloc[-12:], res)
        r2_value = r2_score(filtered_df['y'].iloc[-12:], res)

        # Extract the predicted value at x = 50
        predicted_value_at_50 = res.iloc[-1]  # Last value corresponds to x=50

        metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}, Predicted Value at x=50: {predicted_value_at_50:.2f}"
        future_dates = pd.date_range(start=filtered_df['ds'].max() + pd.DateOffset(months=1), periods=len(future_x),
                                     freq='M')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Prices'))
        fig.add_trace(
            go.Scatter(x=future_dates, y=res, mode='lines+markers', name='Predicted Prices', line=dict(dash='dash')))

        return fig, metrics

    if selected_model == 'ml':
        filtered_df['unique_id'] = 1
        mlf = MLForecast(models=[LinearRegression(), lgb.LGBMRegressor()], lags=[1, 12], freq='M')
        mlf.fit(filtered_df[['unique_id', 'ds', 'y']], static_features=[])
        predictions = mlf.predict(12)

        future_dates = pd.date_range(start=filtered_df['ds'].max() + pd.DateOffset(months=1), periods=len(future_x),
                                     freq='M')
        mse_value = mean_squared_error(filtered_df['y'].iloc[-12:], predictions['LinearRegression'][:12])
        r2_value = r2_score(filtered_df['y'].iloc[-12:], predictions['LinearRegression'][:12])

        # Extract the predicted value at x = 50
        predicted_value_at_50 = predictions['LinearRegression'].iloc[-1]  # Last value corresponds to x=50

        metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}, Predicted Value at x=50: {predicted_value_at_50:.2f}"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Values'))
        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions['LinearRegression'], mode='lines+markers', name='Predicted Values',
                       line=dict(dash='dash')))

        return fig, metrics


app.run_server(debug=True)
