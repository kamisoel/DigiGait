import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_extensions import Download

import numpy as np

from dash_app import utils
from dash_app.figures import *


# COMPONENTS
#===========
def create_header():
    return dbc.Navbar([
        html.A(
                dbc.Row([
                        dbc.Col(html.Img(src=utils.get_asset('app_logo.png'), height="30px")),
                        #dbc.Col(dbc.NavbarBrand("DigiGait", className="ml-2")),
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
        accept='video/*,.c3d',
        children=html.Div([
            'Drag & Drop or ',
            html.A('Click to select'),
            ' a video or c3d file'
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
            marks={i: f'{i}s' for i in range(0, duration+1, max(1,duration//4))},
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

                    dbc.Button("►Advanced options", id='options_btn'),

                    dbc.Collapse(
                        dbc.Col([
                            dbc.Label("Pipeline", html_for="estimator_select"),
                            dbc.Select(
                                id = 'estimator_select',
                                options=[
                                    {"label": "LPN + VideoPose3D", "value": 'lpn'},
                                    {"label": "MediaPipe + VideoPose3D (w/o feet)", "value": 'mp_nf'},
                                    #{"label": "MediaPipe + VideoPose3D (w/ feet)", "value": 'mp_wf', "disabled":True},
                                ],
                                value = 'mp_nf',
                                className="mb-4",
                            ),
                            dbc.Label("Event detection algorithm", html_for="event_detection_select"),
                            dbc.Select(
                                id = 'event_detection_select',
                                options=[
                                    {"label": "Auto", "value": 'auto'},
                                    {"label": "Recurrent Neural Network", "value": 'rnn'},
                                    {"label": "Foot velocity algorithm (FVA)", "value": 'fva'},
                                    {"label": "Foot displacement algorithm", "value": 'simple'},
                                    {"label": "Horizontal Heel Displacement (HHD)", "value": 'hhd'},
                                ],
                                value = 'auto',
                                className="mb-4",
                            ),
                            dbc.Checklist(
                                options=[
                                    {"label": "Debias", "value": "debias", "label_id": "debias"},
                                    {"label": "Skeleton Normalization", "value": "skel_norm", "label_id": "skel_norm"},
                                ],
                                id="option_boxes",
                                className="mb-4",
                                value=['show_cycles','debias'],
                                inline=True,
                                switch=True,
                            ),
                        ]),
                        id="advanced_options",
                        is_open=False,
                    ),
                    dbc.Tooltip(
                        "Correct the systematic underestimation of the angle "
                        "by introducing a scale factor of 1.2",
                        target="debias",
                    ),
                    dbc.Tooltip(
                        "Normalize the bone lengths for close-up recordings",
                        target="skel_norm",
                    ),

                    dbc.Row(
                        dbc.NavLink(
                            dbc.Button("Analyze!", id='analyze_btn',
                                   color='primary', disabled=True),
                            href="#pose_card", external_link=True
                        ), justify='end'),
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
                                utils.get_asset('demo.mp4')),
                        ],
                    ),
                )],
            )
        ]
    )

def overview_settings():
    return html.Div([
        html.Div([
                dbc.Checkbox(
                    id="autoscroll", className="custom-control-input",
                    checked=True
                ),
                dbc.Label(
                    "Autoscroll",  html_for="autoscroll",
                    className="custom-control-label",
                ),
                dbc.Tooltip(
                    "Automatic scroll the graph position to the frame of the 3D viewer",
                    target="autoscroll",
                ),
            ],
            className='custom-switch custom-control custom-control-inline',
        ),
        html.Div([
                dbc.Checkbox(
                    id="show_cycles", className="custom-control-input",
                    checked=True
                ),
                dbc.Label(
                    "Show Cycles",  html_for="show_cycles",
                    className="custom-control-label",
                ),
                dbc.Tooltip(
                    "Show separation lines for the begining of each gait cycle",
                    target="show_cycles",
                ),
            ],
            className='custom-switch custom-control custom-control-inline',
        ),

    ], className='mb-4',)


def pose_card():

    return card(
        id="pose_card",
        header = 'Pose Viewer',
        children=[
            dcc.Loading(
                id="pose-loading",
                children=[
                    dbc.Row([
                        dcc.Store(id='pose_data'),
                        dbc.Toast(
                            [html.P("Could not estimate pose correctly!", className="mb-0")],
                            id="error-toast",
                            header="Something went wrong :(",
                            icon="danger",
                            is_open=False,
                            dismissable=True,
                            style={"position": "fixed", "top": 60, "right": 10, "width": 350},
                        ),
                        dbc.Col(
                            # 3d viewer
                            dcc.Graph(
                                id="pose_graph",
                                figure=create_skeleton_fig(demo_pose, eye=eye),
                                config={'displaylogo': False,},
                            ), md=5),
                        dbc.Col(
                            dbc.Tabs([
                                dbc.Tab(
                                    html.Div([
                                        dcc.Graph(
                                            id="angle_graph",
                                            figure=create_angle_figure(demo_angles, demo_events[0]),
                                            config={'displaylogo': False,
                                                    'modeBarButtonsToAdd': ['drawline',
                                                                            'drawcircle',
                                                                            'drawrect',
                                                                            'eraseshape'],
                                                   'scrollZoom':True},
                                        ),
                                        overview_settings(),
                                    ]), label='Overview',
                                ),
                                dbc.Tab(
                                    dcc.Graph(
                                        id="gait_phase_graph",
                                        figure=create_gait_phase_figure(demo_avg, norm_data),
                                        config={'displaylogo': False,
                                                'modeBarButtonsToAdd': ['drawline',
                                                                        'drawcircle',
                                                                        'drawrect',
                                                                        'eraseshape'],
                                               'scrollZoom':True},
                                    ), label='Avg. gait phase'
                                ),
                                dcc.Tab(
                                    dbc.Table.from_dataframe(metrics, striped=True, bordered=True, 
                                        dark=True, responsive=True,),
                                    id = 'metrics',
                                    label='Metrics',
                                ),
                                dcc.Tab([
                                    dcc.Graph(
                                        id='phase_space_reconstruction',
                                        figure=create_phase_space_reconstruction(emb),
                                        config={'displaylogo': False,
                                                'modeBarButtonsToAdd': ['drawline',
                                                                        'drawcircle',
                                                                        'drawrect',
                                                                        'eraseshape'],
                                               'scrollZoom':True}
                                    ),
                                    html.Div(
                                        dbc.Table.from_dataframe(nl_metrics, striped=True, bordered=True, 
                                                                dark=True, responsive=True),
                                        id = 'nl_metrics'
                                    )
                                ], label='Nonlinear')
                            ]), md=7)
                    ]),

                    dbc.Row([
                        #dbc.Col(
                        #    dbc.Button("Export as .csv", color="primary", id='csv_download_btn'),
                        #    width='auto'),
                        dbc.Col(
                            dbc.Button("Save as .c3d", color="primary", id='c3d_download_btn'),
                            width='auto')
                    ], justify="end")
                ],
            )
        ]
    )

# INIT STATE
#=======
demo_pose, demo_angles, demo_events = utils.get_demo_data()
demo_avg = utils.avg_gait_phase(demo_angles, demo_events)
norm_data = utils.get_norm_data()['KneeZ']
metrics = utils.calc_metrics(demo_angles, demo_events)
emb, nl_metrics = utils.calc_nonlinear(demo_angles)
eye = utils.get_sagital_view(demo_pose)

# LAYOUT
#=======
layout = html.Div([
    dcc.Store(id='video_data'),
    dcc.Store(id='pose_data_video'),
    dcc.Store(id='pose_data_c3d'),
    dcc.Interval(id='animator', interval=500, disabled=False),
    Download(id="download"),
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
