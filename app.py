import flask
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
#import dash_uploader as du

import numpy as np

from pathlib import Path
import subprocess
import base64
import tempfile
import uuid

external_stylesheets = [dbc.themes.BOOTSTRAP]

# Server definition

server = flask.Flask(__name__)
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                server=server)

# HEADER
# ======
header = dbc.Navbar([
        html.A(
            dbc.Row([
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Gait Analyzer", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="#",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav([
                #dbc.NavItem(button_github),
            ], navbar=True), 
            id="navbar-collapse", 
            navbar=True
        ),
    ],
    color="dark",
    dark=True,
    sticky="top",
    className="mb-4",
)

# FIGURES
# =======
def frame_args(duration, redraw=True, transition=False):
    return {
            "frame": {"duration": duration, "redraw": redraw},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 0},
        }

def create_skeleton_fig(pose_3d, skeleton, joints, fps=25, height=500):

    lines = {'frame': [], 'joint': [], 'x':[], 'y':[], 'z':[]}
    for f in range(len(pose_3d)):
        for j in range(len(joints)):
            p = skeleton[j]
            if p != -1:
                lines['frame'].extend([f]*3)
                lines['joint'].extend([joints[j], joints[p], None])
                for i, c in enumerate(list('xyz')):
                    lines[c].append(pose_3d[f, j, i])
                    lines[c].append(pose_3d[f, p, i])
                    lines[c].append(None)
    pose_df = pd.DataFrame.from_dict(lines)
    
    # Create figure
    frames = [go.Frame(
            name=str(frame),
            data=[go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                    mode='markers+lines', line=dict(width=5),
                    marker=dict(size=5),
                    hovertemplate= '<b>%{text}</b><br>'+
                                   '<b>x</b>: %{x:.3f}<br>'+
                                   '<b>y</b>: %{y:.3f}<br>'+
                                   '<b>z</b>: %{z:.3f}<br>'+
                                   '<extra></extra>',
                    text = df['joint']
            )])
            for frame, df in pose_df.groupby('frame')]
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 15},
            "prefix": "Frame:",
            "xanchor": "right"
        },
        "pad": {"b": 10, "t": 15},
        "len": 0.7,
        "x": 0.25,
        "y": 0,
        "steps": [{
            "args": [
                [frame], frame_args(0)
            ],
            "label": frame,
            "method": "animate"}
        for frame in range(0, len(pose_3d), fps)]
    }

    layout=go.Layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)', # transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0), # tight layout
        scene = go.layout.Scene( # scene dimension
            xaxis=dict(range=[-.75,.75], autorange=False, zeroline=False),
            yaxis=dict(range=[-.75,.75], autorange=False, zeroline=False),
            zaxis=dict(range=[-0.2, 2], autorange=False, zeroline=False),
            aspectratio=dict(x=1, y=1, z=2.),
        ),
        scene_camera=dict(
            eye=dict(x=-1.0, y=3.0, z=.5),
        ),
        hovermode="closest",
        height=height, #width=400,
        sliders=[sliders_dict],
        updatemenus=[{
            "buttons":[{
                        "args": [None, frame_args(1./fps)],
                        "label": "&#9654;", # play symbol
                        "method": "animate"
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate"
                    }],
            "direction": "left",
            "pad": {"r": 10, "t": 40},
            "showactive": False,
            "type": "buttons",
            "x": 0,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top"
        }]
    )

    return go.Figure(data=frames[0].data, layout=layout, frames=frames)

def create_angle_figure(angles, joint='knee_flex'):
    names = ['Right', 'Left']
    fig = go.Figure()#make_subplots(2, 1, shared_xaxes=True)
    for i, n in enumerate(['r_'+joint, 'l_'+joint]):
        fig.add_trace(
            go.Scatter(
                y=angles[n],
                name=names[i], meta=names[i],
                hovertemplate= '%{meta}: %{y:.1f}°'+
                                '<extra></extra>'
            )#, i+1, 1
        )
    #fig.update_yaxes(matches='y')
    fig.update_layout(
        dragmode= 'pan', 
        xaxis=dict(range=[0,300]), 
        yaxis=dict(fixedrange=True),
        margin=dict(l=10, r=10, b=10, t=10),
        hovermode="x unified",
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hoverlabel_bgcolor='black'
    )
    fig.add_shape(
        dict(type="line", x0=0, x1=0, y0=120, y1=200, line_color="green"), 
        #row="all", col=1
    )
    return fig


# COMPONENTS
# ==========

def optional(*elems):
    """Small helper to ignore Nones in a List"""
    return [e for e in elems if e is not None]

def card(id, header="", children=[]):
    return dbc.Card(
        id = id,
        color="dark",
        children=optional(
            dbc.CardHeader(header) if header else None,
            dbc.CardBody(children),
        )
    )

def get_upload_component(id):
    return dcc.Upload(
        id=id,
        className='upload',
        accept='video/*',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File'),
            ' to change video'
        ]),
    )

def get_video_player(id, src):
    return html.Video(
        id=id,
        controls=True, 
        width='100%',
        src=src
    )

def video_range_slider(duration):
    if not duration:
        duration=20
    return [
        dbc.Label("Video Range", html_for="slider"),
        dcc.RangeSlider(
            id='video_range',
            min=0,
            max=duration,
            marks={i: f'{i}s' for i in range(0, duration+1, duration//4)},
            pushable=3,
            value=[0,duration]
        ),
    ]
    

# Video settings
def video_settings():
    return card(
        id = 'video_settings',
        header = 'Choose a video to analyze',
        children=[
            dbc.Row([
                dbc.Col([
                    get_upload_component(id='video_uploader'),
                    
                    dbc.FormGroup(
                        video_range_slider(15),
                        id='video_range_group',
                        className="mb-4",
                    ),

                    dbc.Checklist(
                        options=[
                            {"label": "Show Detections", "value": 1},
                        ],
                        id="show-detections",
                        className="mb-4",
                        value=[],
                        #switch=True,
                    ),
                    
                    dbc.Button('Analyze!', color='primary', id='analyze_btn'),
                ], className="mb-4",)
            ]),
        ],
    )

def video_preview():
    return card(
        id = 'video_preview',
        header = 'Video preview',
        children=[
            dbc.Row([
                dbc.Col(
                    id='video_container',
                    children=[
                        get_video_player(
                            'video_player',
                            app.get_asset_url('video-clip.mp4')),
                    ],
                )],
            )
        ]
    )


def pose_card():
    return card(
        id="pose-card",
        header = 'Pose Viewer',
        children=[
            dcc.Loading(
                id="pose-loading",
                type="circle",
                children=dbc.Row([
                    dbc.Col(
                        # 3d viewer
                        dcc.Graph(
                            id="pose_graph",
                            figure=create_skeleton_fig(pose_3d, skeleton, joints),
                            config={'displaylogo': False,},
                        ), md=5),
                    dbc.Col(
                        # knee joint angle
                        dcc.Graph(
                            id="angle_graph",
                            figure=create_angle_figure(angles),
                            config={'displaylogo': False,
                                   'scrollZoom':True},
                        ), md=7)
                ]),
            )
        ]
    )


# INTERACTION
# ===========

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    return not is_open if n else is_open

# use ffprobe to get the duration of a video
def get_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

@app.callback(Output('video_container', 'children'),
              Output('video_range_group', 'children'),
              Input('video_uploader', 'contents'),
              State('video_uploader', 'filename'),
              State('video_uploader', 'last_modified'))
def update_output(content, name, date):
    if content is None:
        raise PreventUpdate
    file = UPLOAD_ROOT / name
    content_type, content_string = content.split(',')
    file.write_bytes(base64.b64decode(content_string))
    duration = int(get_duration(file))
    slider = video_range_slider(duration)
    player = get_video_player(id='video_player', src=content)
    return player, slider



# APP LAYOUT
# ==========


app.layout = html.Div([
    header,
    dbc.Container([
        dbc.Row([
                dbc.Col(video_settings(), md=5),
                dbc.Col(video_preview(), md=7),
            ],
            className="mb-4",
        ),
        dbc.Row([
                dbc.Col(pose_card(), md=12),
            ],
            className="mb-4",
        )
    ],
    fluid=True,)
])

if __name__ == '__main__':
    app.run_server(debug=True)