import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import sqlalchemy
from datetime import datetime, timedelta
import plotly.graph_objs as go
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
import lightgbm as lgb
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

username = 'root'
password = 'sayan123'
host = 'localhost'
database = 'mcd'


# SQLAlchemy connection string
connection_string = f"mysql+pymysql://{username}:{password}@{host}/{database}"

# Create a database engine
engine = sqlalchemy.create_engine(connection_string)
query = "SELECT Product, Date, Price FROM commodities where Date > '2018-04-01'"
df = pd.read_sql(query, engine)

df['Date'] = pd.to_datetime(df['Date'])
df['date_int'] = (df['Date'] - df['Date'].min()) / timedelta(days=1)
min_date_int = df['date_int'].min()
max_date_int = df['date_int'].max()



# Initialize Dash app
app = dash.Dash("Demo")

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': i, 'value': i} for i in df['Product'].unique()],
        value=df['Product'].unique()[0]
    ),

    dcc.RangeSlider(
        id='date-range-slider',
        min=min_date_int,
        max=max_date_int,
        step=1,
        value=[min_date_int, max_date_int],
        marks={int(date_int): {'label': str(date), 'style': {'transform': 'rotate(-45deg)'}}
               for date_int, date in zip(range(int(min_date_int), int(max_date_int), 30),
                                         pd.date_range(df['Date'].min(), df['Date'].max(), freq='ME'))}
    ),

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
    [Input('product-dropdown', 'value'),
     Input('date-range-slider', 'value'),
     Input('model-dropdown', 'value')]
)
def update_graph(selected_product, date_range_slider_values, selected_model):
    # Convert slider values back to dates
    start_date = df['Date'].min() + timedelta(days=date_range_slider_values[0])
    end_date = df['Date'].min() + timedelta(days=date_range_slider_values[1])

    filtered_df_product = df[(df['Product'] == selected_product)]
    filtered_df = filtered_df_product[
        (df['Date'] >= start_date) & (df['Date'] <= end_date)]

    filtered_df = filtered_df.sort_values('Date')
    filtered_df['Price'].fillna(0, inplace=True)
    future_dates = pd.date_range(start=filtered_df['Date'].max(), periods=12, freq="ME")
    filtered_df.rename(columns={"Date": "ds", "Product": "unique_id", "Price": "y"}, inplace=True)

    if selected_model == 'stat':
        from statsforecast.models import AutoARIMA

        # Create an integer representation of the date for the exogenous feature
        filtered_df['ds'] = pd.to_datetime(filtered_df['ds'])
        filtered_df['date_int'] = (filtered_df['ds'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        filtered_df["unique_id"] = 1

        # Instantiate the ARIMA model
        models = [AutoARIMA()]

        # Initialize StatsForecast
        sf = StatsForecast(models=models, freq='M')

        # Fit the model using the exogenous feature (date_int)
        sf.fit(filtered_df)

        # Prepare the future dates as exogenous features for prediction
        future_dates = pd.date_range(start=filtered_df['ds'].max(), periods=13, freq='M')[1:]
        future_dates_int = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        X_df = pd.DataFrame({'date_int': future_dates_int})
        X_df['unique_id'] = 1
        X_df['ds'] = future_dates

        # Generate predictions for the next 12 periods with the exogenous feature
        predictions = sf.predict(h=12,X_df=X_df)

        mse_value = mean_squared_error(filtered_df['y'].iloc[-12:], predictions['AutoARIMA'][:12])
        r2_value = r2_score(filtered_df['y'].iloc[-12:], predictions['AutoARIMA'][:12])
        metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}"

        # Create a plotly figure
        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Prices'))

        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions['AutoARIMA'], mode='lines+markers', name='Predicted Prices',
                       line=dict(dash='dash')))

        # Highlight the forecast range using a vertical rectangle
        fig.add_vrect(x0=start_date, x1=end_date, fillcolor="LightSkyBlue", opacity=0.5, layer="below", line_width=0)

        return fig,metrics

    if selected_model == 'dl':
        filtered_df['ds'] = pd.to_datetime(filtered_df['ds'])
        horizon = 12  # len(test)
        models = [NBEATS(input_size=2 * horizon, h=horizon, max_steps=10),
                  NHITS(input_size=2 * horizon, h=horizon, max_steps=10)]

        # Specify static_features=[] to indicate all features are dynamic
        nf = NeuralForecast(models=models, freq='M')

        nf.fit(df=filtered_df)
        predictions = nf.predict().reset_index()
        res = (predictions['NHITS'] + predictions['NBEATS']) / 2
        mse_value = mean_squared_error(filtered_df['y'].iloc[-12:], res)
        r2_value = r2_score(filtered_df['y'].iloc[-12:], res)
        metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Prices'))
        fig.add_trace(
            go.Scatter(x=future_dates, y=res, mode='lines+markers', name='Predicted Prices',
                       line=dict(dash='dash')))
        fig.add_vrect(
            x0=start_date, x1=end_date,
            fillcolor="LightSkyBlue", opacity=0.5,
            layer="below", line_width=0,
        )
        return fig,metrics

    if selected_model == 'ml':
         # Assuming filtered_df is already defined
         filtered_df['ds'] = pd.to_datetime(filtered_df['ds'])
         filtered_df['date_int'] = (filtered_df['ds'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
         filtered_df['unique_id'] = 1   #range(1, len(filtered_df) + 1)
         mlf = MLForecast(
             models=[LinearRegression(), lgb.LGBMRegressor()],
             lags=[1, 12],  # Lag features
             freq='M'  # Monthly frequency
         )

         # Fit the model
         mlf.fit(filtered_df[['unique_id','ds', 'y']], static_features=[])

         # Get predictions for 12 future periods
         predictions = mlf.predict(12)

         # Generate future dates for plotting
         future_dates = pd.date_range(start=filtered_df['ds'].max(), periods=12, freq='M')

         mse_value = mean_squared_error(filtered_df['y'].iloc[-12:], predictions['LinearRegression'][:12])
         r2_value = r2_score(filtered_df['y'].iloc[-12:], predictions['LinearRegression'][:12])
         metrics = f"MSE: {mse_value:.2f}, R²: {r2_value:.2f}"

         # Plot the actual vs predicted values
         fig = go.Figure()

         # Add actual data
         fig.add_trace(
             go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines+markers', name='Actual Prices')
         )

         # Add predicted data for Linear Regression
         fig.add_trace(
             go.Scatter(x=future_dates, y=predictions['LinearRegression'], mode='lines+markers',
                        name='Predicted Prices', line=dict(dash='dash'))
         )

         # Mark a vertical rectangle (if you want to highlight a specific period)

         fig.add_vrect(
             x0=start_date, x1=end_date,
             fillcolor="LightSkyBlue", opacity=0.5,
             layer="below", line_width=0,
         )

         return fig,metrics

app.run_server(debug=True)
