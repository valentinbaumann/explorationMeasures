
import plotly.graph_objects as go

def plotPath(x,y,fig=None):
    """Returns figure obj plotting a single 2d path. 3d path required as input."""
    if fig is None:
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )    
    return fig

def plotPathSet(data, df_obj = None, samples=256):
    """Returns figure obj plotting multiple 2d paths. 3d path required as inout. """

    if not df_obj is None:
        fig = plotObjectLocations(df_obj)
    else:
        fig = go.Figure()
    for path in data:
        if not samples is None:
            steps = max(int(len(path[:,0])/samples),1)
            fig = plotPath(path[::steps,0],path[::steps,2], fig)
        else:
            fig = plotPath(path[:,0],path[:,2], fig)
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )
    return fig



def plotObjectLocations(df_obj):
    """Returns figure obj containing scatterplot displaying all obj positions. Requires a dataframe with two columns called X and Z as input """

    # https://stackoverflow.com/questions/61342459/how-can-i-add-text-labels-to-a-plotly-scatter-plot-in-python
    layout = dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'))

    data = go.Scatter(x=df_obj['X'],
                    y=df_obj['Z'],
                    text=df_obj['Landmark'],
                    textposition='top right',
                    mode='markers+text',
                    name='Objects')

    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )
    return fig



def plotHeatmap(heatmap, plotTitle = ""):
    """Returns a heatmap with uniplor scale (0 to n). Requires a 2d numpy array representing the heatmap histogram as input."""

    fig = go.Figure(data=go.Heatmap(
                z=heatmap, colorscale = "Blues")) # good uniploar scales: "hot", "blues", "reds"
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )  
    fig.update_layout(title = plotTitle)
    return fig



def plotDiffHeatmap(heatmap, plotTitle = ""):
    """Returns a heatmap with bipolar scale (-n to n). Requires a 2d numpy array representing the heatmap histogram as input.
    If this is used to compare to heatmaps, input should be the difference array for the two to be compared heatmaps."""
    fig = go.Figure(data=go.Heatmap(
                z=heatmap, zmid = 0, colorscale = "Picnic")) # good bipolar scales: "picnic", "earth"
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )
    fig.update_layout(title = plotTitle)      
    return fig

    







