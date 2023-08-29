# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import car_data_dealer
import plotly.express as px
import feffery_antd_components as fac  # 通用组件库
from dash.dependencies import Input, Output, State
import cleaned_dataset
# Incorporate data
# df = pd.read_csv('E:\MachineLearning\dataCars.csv')
# test_df = pd.read_csv('D:\\学习资料\\项目\\testCars.csv')

original_df = pd.read_csv('E:\MachineLearning\data\Cars.csv')

# Initialize the app
app = Dash(__name__)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# 在这里写操作说明
operator_describe = '''
                    "show" will display all the data saved in the path of the local database you entered;
                    "predict" is about to show the price of the car under the path data that you have entered into a local database
'''
# App layout
app.layout = html.Div([
    html.H1(
        children='Hello Dash',
        style={
            'margin-top': '6%',
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children=operator_describe, style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    # 放置水平分割线
    fac.AntdDivider(isDashed=True),

    fac.AntdRow(
        [
            fac.AntdCol(
                html.Div(
                ),
                span=6
            ),
            fac.AntdCol(
                html.Div(
                    [
                        fac.AntdInput(
                            placeholder='please input training url',
                            id="training_url",
                            style={'width': 230}
                        ),
                        fac.AntdButton(
                            'show',
                            id="training_btn",
                            type='primary',
                            icon=fac.AntdIcon(icon='md-layers')
                        ),
                    ],
                    style={
                        'color': 'white',
                        'height': '100px',
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center'
                    }

                ),
                span=6,

            ),
            fac.AntdCol(
                html.Div(
                    [
                        fac.AntdInput(
                            placeholder='please input predict url',
                            id="predict_url",
                            style={'width': 230}
                        ),
                        fac.AntdButton(
                            'predict',
                            id="predict_btn",
                            type='primary',
                            icon=fac.AntdIcon(icon='fc-repair')
                        ),
                    ],
                    style={
                        'color': 'white',
                        'height': '100px',
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center'
                    }
                ),
                span=6
            ),
            fac.AntdCol(
                html.Div(
                ),
                span=6
            ),
        ],
        gutter=10
    ),

    # 放置水平分割线
    fac.AntdDivider(isDashed=True),
    html.H1(children='Your Csv Data'),
    html.Div(id='training_dataset'),
    fac.AntdDivider(isDashed=True),
    html.H1(children="Your Predict Data . The Columns 'selling_price' Is  Predict Result"),
    html.Div(id='predict_table'),


])


@app.callback(
    Output('training_dataset', 'children'),
    Input('training_btn', 'nClicks'),
    State('training_url', 'value'),
)
def button_click_training_data(nClicks, input_value):
    # 'D:\\学习资料\\项目\\Cars.csv'
    print(input_value)
    if input_value != None:
        # 显示数据表格
        df = pd.read_csv(input_value)
        # df = pd.read_csv('D:\\学习资料\\项目\\Cars.csv')
        _df = car_data_dealer.cleaned_dataset(df.copy())
        return dash_table.DataTable(data=_df.to_dict('records'), page_size=10,

                                    style_cell={
                                        'minWidth': '130px',
                                        'width': '130px',
                                        'maxWidth': '130px',
                                        'height': 'auto'},
                                    )
    else:
        return fac.AntdAlert(
            message='please input a file url'
        )


@app.callback(
     Output('predict_table', 'children'),
     Input('predict_btn', 'nClicks'),
     State('predict_url', 'value')
)
def button_click_pre_data(nClicks, input_value):
    # 'D:\\学习资料\\项目\\testCars.csv'
    print(input_value)
    if input_value != None:
        # 显示数据表格
        test_df = pd.read_csv(input_value)
        # df = pd.read_csv('D:\\学习资料\\项目\\Cars.csv')
        _df = cleaned_dataset.predict_price(original_df,test_df)
        return dash_table.DataTable(data=_df.to_dict('records'), page_size=10,
                                    columns=[{'id': c, 'name': c} for c in _df.columns],
                                    style_cell_conditional=[{
                                        'if': {'column_id': 'selling_price'},
                                        'backgroundColor': 'rgb(204, 255, 255)',
                                        'color': 'black'
                                    }],
                                    style_cell={
                                        'minWidth': '130px',
                                        'width': '130px',
                                        'maxWidth': '130px',
                                        'height': 'auto'},
                                    )
    else:
        return fac.AntdAlert(
            message='please input a file url'
        )


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

