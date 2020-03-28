import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output

pred = np.load('test_predict.npy')
truth = np.load('test_truth.npy')


def create_graph(crime, real):
    print(real)
    figure = go.Figure()

    obj = html.Div([
            html.Div([
            html.H4(
                children=real,
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),
            dcc.Graph(
                id=real+str(crime),
                figure=figure
            )
        ], className="col-sm")
        ], className="row")

    return obj


def make_row(crime):
    return html.Div(
        children=[
            create_graph(crime, 'Truth'), 
            create_graph(crime, 'Predicted'),         
            html.Div([
                dcc.Slider(
                id='slider'+str(crime),
                min=0,
                max=15,
                step=1,
                value=0,
                )
            ], className="row")]
        )

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']
app = dash.Dash(external_stylesheets=external_stylesheets)
# app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    style={'backgroundColor': colors['background']}, 
    children=[

        html.H1(
            children='Robberies at time 10',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div(children='Dash: A web application framework for Python.', style={
            'textAlign': 'center',
            'color': colors['text']
        }),

        *[make_row(i) for i in range(5)],

    ], className="container")

@app.callback(Output('Truth0', 'figure'),
              [Input('slider0', 'value')])
def truth0(value):
    data = truth
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 0], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Truth1', 'figure'),
              [Input('slider1', 'value')])
def truth1(value):
    data = truth
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 1], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Truth2', 'figure'),
              [Input('slider2', 'value')])
def truth2(value):
    data = truth
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 2], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Truth3', 'figure'),
              [Input('slider3', 'value')])
def truth3(value):
    data = truth
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 3], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Truth4', 'figure'),
              [Input('slider4', 'value')])
def truth4(value):
    data = truth
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 4], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Predicted0', 'figure'),
              [Input('slider0', 'value')])
def pred0(value):
    data = pred
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 0], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Predicted1', 'figure'),
              [Input('slider1', 'value')])
def pred1(value):
    data = pred
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 1], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Predicted2', 'figure'),
              [Input('slider2', 'value')])
def pred2(value):
    data = pred
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 2], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Predicted3', 'figure'),
              [Input('slider3', 'value')])
def pred3(value):
    data = pred
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 3], hoverinfo='skip'))
    return {'data': traces}

@app.callback(Output('Predicted4', 'figure'),
              [Input('slider4', 'value')])
def pred4(value):
    data = pred
    traces = []
    for i in range(data.shape[0]):
        traces.append(go.Heatmap(z=data[value, :, :, 4], hoverinfo='skip'))
    return {'data': traces}


if __name__ == '__main__':
    app.run_server(debug=True)



# {
#                 'data': [
                    
#                 ],

#             }