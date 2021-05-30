#from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Output, Input, State, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate

from dash_app.layout import video_range_slider, get_video_player
from dash_app import utils
from dash_app.config import Config
from dash_app import figures

import base64
from pathlib import Path
import json

def register_callbacks(app):
    # add callback for toggling the collapse on small screens
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        return not is_open if n else is_open


    @app.callback(Output('video_container', 'children'),
                  Output('video_range_group', 'children'),
                  ServersideOutput('video_data', 'data'),
                  Input('video_uploader', 'contents'),
                  State('video_uploader', 'filename'),
                  State('video_uploader', 'last_modified'),)
                  #State('session', 'data'))
    def update_output(content, name, date):
        #if session_data is not None and 'upload_dir' in session_data:
        #    upload_dir = Path(session_data['upload_dir'])
        #else:
        #    upload_dir = utils.random_upload_url(mkdir=True)
        #    session_data = {'upload_dir': str(upload_dir)}
        #file = upload_dir / 'video.mp4'
        # TODO: video file conversion
        content_type, content_string = content.split(',')
        #file.write_bytes(base64.b64decode(content_string))
        video_content = base64.b64decode(content_string)
        duration = round(utils.get_duration(utils.memory_file(video_content)))

        slider = video_range_slider(duration)
        player = get_video_player(id='video_player', src=content)
        return player, slider, video_content


    @app.callback(Output('analyze_btn', 'disabled'),
                  Input('video_data', 'modified_timestamp'))
    def activate_btn(t):
        return False


    @app.callback(#Output('skel_graph', 'figure'),
                  Output('console', 'children'),
                  Input('angle_graph', 'clickData'))
    def show_frame(click_data):
        print(click_data)
        return json.dumps(click_data, indent=2)


    @app.callback(Output('pose_graph', 'figure'),
                  Output('angle_graph', 'figure'),
                  Trigger('analyze_btn', 'n_clicks'),
                  State('video_data', 'data'),
                  State('video_range', 'value'),
                  State('estimtor_choice', 'label'),
                  )
    def analyze_clicked(video_content, slider_value, pipeline):
        #upload_dir = session_data['upload_dir']
        video_path = utils.memory_file(video_content)
        pose_3d, knee_angles = utils.run_estimation(video_path, slider_value, pipeline)
        skel_fig = figures.create_skeleton_fig(pose_3d)
        ang_fig = figures.create_angle_figure(knee_angles)
        return skel_fig, ang_fig

