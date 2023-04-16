import io
import os
import base64

import pandas as pd
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
from configargparse import ArgumentParser

import numpy as np

# This is done specifically for the set stl10 ran on 2 gpus.
# More general plotting can be used for any number of gpus and sets, by changing few parameters

# load saved data
data = np.load("Data/pleasework.npz")

# load each array ( making sure they are of correct type
tx = data['tx']
ty = data['ty']
paths = data['path_bank'].astype(int)
labels = data['label_bank'].astype(int)

# two definitions of class names based on ordering
# this is defined in labels as 10, 1, 2, 3, 4, 5, 6, 7, 8, 9
# i.e. for indexing convert 10 to 0
class_names = ["truck", "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship"]
# as the files are ordered alphabeticaly, the order the testing on the classes is executed is
# 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, i.e., in array labels, 0 corresponds to 1 and 1 corresponds to 10
class_nm = ["airplane", "truck", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship"]
labels_real = [class_nm[label] for label in labels]

# converting numpy arrays to pandas datasets
# this way we have legend which can isolate classes for closer inspection
d = {"tx": tx, "ty": ty, "colors": labels_real}
df = pd.DataFrame(d)

# create scatter plot with legend
fig = px.scatter(df, x="tx", y="ty", color="colors")
fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
    showlegend=True)

fig.update_layout(
    xaxis=dict(range=[-1.1, 1.1]),
    yaxis=dict(range=[-1.1, 1.1]),
    width=1000, height=1000,
    plot_bgcolor="white",
    yaxis_showticklabels=False,
    xaxis_showticklabels=False,
    yaxis_visible=False,
    xaxis_visible=False,)

# fig.show()
# Set up the app now
app = Dash(__name__)

app.layout = html.Div(
    className="container",
    children=[dcc.Graph(id="graph-2-dcc", figure=fig, clear_on_unhover=True),
              dcc.Tooltip(id="graph-tooltip-2", direction='bottom')])

@app.callback(
    Output("graph-tooltip-2", "show"),
    Output("graph-tooltip-2", "bbox"),
    Output("graph-tooltip-2", "children"),
    Output("graph-tooltip-2", "direction"),
    Input("graph-2-dcc", "hoverData")
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update, no_update
    im_index = hoverData['points'][0]['pointIndex']
    # this is caused by the use of 2 gpus, if only one is used we can have:
    # p = paths[im_index + 800*hoverData['points'][0]['curveNumber']]
    if im_index > 400:
        p = paths[im_index + 400*hoverData['points'][0]['curveNumber'] + 3600]
    else:
        p = paths[im_index + 400*hoverData['points'][0]['curveNumber']]

    img_name = p[1]
    class_id = p[0]
    # create path from img_name and class_name
    image_path = 'Data/test/' + str(class_id) + '/' + str(img_name) + '.png'
    im = Image.open(image_path)

    # dump it to base64
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]

    # control the position of the tooltip
    y = hover_data["y"]
    direction = "bottom" if y > 0.5 else "top"

    if class_id == 10:
        class_id = 0
    children = [
        html.Img(
            src=im_url,
            style={"width": "150px"},
        ),
        html.P(img_name),
        html.P(class_names[int(class_id)])
    ]
    return True, bbox, children, direction


if __name__ == "__main__":
    app.run_server(debug=True)
