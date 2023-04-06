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


# data = np.load("Data/plot_data.npz")
data = np.load("Data/s_plot_data.npz")

tx = data['tx']
ty = data['ty']
paths = data['path_bank'].flatten()
labels = data['label_bank']

for i in range(len(paths)):
    index = paths[i].find('Data')
    paths[i] = paths[i][index:]


'''color_map = {
    0: "#006400", #dark green
    1: "#00008B", #dark blue
    2: "#B03060", #maroon3
    3: "#FFFF00", #yellow
    4: "#FF0000", #red
    5: "#DEB887", #brown
    6: "#0FF000", #lime
    7: "#00FFFF", #blue
    8: "#FF00FF", #pink
    9: "#6495ED", #light blue
    10: "#000000"}#black'''

class_names = ["truck", "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship"]
#colors = [color_map[label] for label in labels]
clas = ["airplane", "truck", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship"]
labels_real = [clas[label] for label in labels]
d = {"tx": tx, "ty": ty, "colors": labels_real}
df = pd.DataFrame(d)


fig = px.scatter(df, x="tx", y="ty", color="colors")
fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
    #marker=dict(size=5, color=colors),
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
    # Load image with pillow
    # image_path = '/home/msl/Pictures/background.jpg'
    image_path = paths[hoverData['points'][0]['pointIndex'] + 800*hoverData['points'][0]['curveNumber']]
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

    img_name = image_path.split('/')[-1]
    class_id = image_path.split('/')[-2]
    if class_id == "10":
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
