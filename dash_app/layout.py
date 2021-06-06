import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from dash_app.utils import get_asset, get_demo_data
from dash_app.figures import create_skeleton_fig, create_angle_figure


# COMPONENTS
#===========
def create_header():
    return dbc.Navbar([
        html.A(
                dbc.Row([
                        dbc.Col(html.Img(src=get_asset('logo.png'), height="30px")),
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
            tooltip = { 'always_visible': True, 'placement': 'bottom'},
            pushable=3,
            value=[0,duration],
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
                        video_range_slider(20),
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
                        inline=True,
                        switch=True,
                    ),
                    dbc.Select(
                        id = 'estimator_select',
                        options=[
                            {"label": "LPN + VideoPose3D", "value": 0},
                            {"label": "MediaPipe + VideoPose3D (w/o feet)", "value": 1},
                            {"label": "MediaPipe + VideoPose3D (w/ feet)", "value": 2, "disabled":True},
                        ],
                        value = 1,
                        className="mb-4",
                    ),
                    dbc.NavLink(
                        dbc.Button("Analyze!", id='analyze_btn',
                               color='primary', disabled=True),
                        href="#pose_card", external_link=True
                    ),
                    html.Pre(id='console')
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
                    dcc.Loading(
                        id='video_container',
                        children=[
                            get_video_player(
                                'video_player',
                                get_asset('demo.mp4')),
                        ],
                    ),
                )],
            )
        ]
    )


def pose_card():

    return card(
        id="pose_card",
        header = 'Pose Viewer',
        children=[
            dcc.Loading(
                id="pose-loading",
                children=dbc.Row([
                    dbc.Col(
                        # 3d viewer
                        dcc.Graph(
                            id="pose_graph",
                            figure=create_skeleton_fig(demo_pose),
                            config={'displaylogo': False,},
                        ), md=5),
                    dbc.Col(
                        # knee joint angle
                        dcc.Graph(
                            id="angle_graph",
                            figure=create_angle_figure(demo_angles),
                            config={'displaylogo': False,
                                   'scrollZoom':True},
                        ), md=7)
                ]),
            )
        ]
    )

# LAYOUT
#=======
demo_pose, demo_angles = get_demo_data()

layout = html.Div([
    #dcc.Store(id='session', storage_type='session'),
    dcc.Store(id='video_data'),
    create_header(),
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
