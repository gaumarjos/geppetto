





def update_map3(selected_points):
    selected_indexes = []
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]

    if 0:
        if len(selected_indexes) > 0:
            fig = px.scatter(x=my_df.iloc[selected_indexes]['lon'],
                             y=my_df.iloc[selected_indexes]['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
        else:
            fig = px.scatter(x=my_df['lon'],
                             y=my_df['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
    else:
        if len(selected_indexes) > 0:
            print("should be partial")
            print(selected_indexes)
            data = [go.Scattermapbox(lat=my_df.iloc[selected_indexes]['lat'],
                                     lon=my_df.iloc[selected_indexes]['lon'],
                                     mode='markers',
                                     marker=dict(
                                         size=8,
                                         color='yellow',
                                         opacity=1.0)
                                     )]
        else:
            data = [go.Scattermapbox(lat=my_df['lat'],
                                     lon=my_df['lon'],
                                     mode='lines',
                                     line=dict(
                                         width=2,
                                         color='red'),
                                     hovertext=my_df['c_dist_geo2d'],
                                     )]

        layout = go.Layout(autosize=False,
                           mapbox=dict(bearing=0,
                                       pitch=0,
                                       zoom=10,
                                       center=go.layout.mapbox.Center(
                                           lon=np.mean(my_df["lon"]),
                                           lat=np.mean(my_df["lat"])
                                       ),
                                       # accesstoken=map_token,
                                       style="open-street-map"),
                           margin={'l': 10, 'b': 10, 't': 10, 'r': 10},
                           hovermode='closest'
                           )

        fig = go.Figure(data=data, layout=layout)

    # fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    # fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    # fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    return fig