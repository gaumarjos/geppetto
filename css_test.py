from dash import Dash, html, dcc

app = Dash(__name__)

app.layout = html.Div([
    html.Div(
        className="app-header",
        children=[
            html.Div('Plotly Dash', className="app-header--title")
        ]
    ),
    html.Div(
        className="prova",
        children=[
            html.Div(
                [
                    html.H5('Overview'),
                    html.Div("testo di prova")
                ],
            ),
        ],
    ),

])

if __name__ == '__main__':
    app.run_server(debug=True)
