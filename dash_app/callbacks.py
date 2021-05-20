from dash.dependencies import Input, Output, State
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
                  Output('session', 'data'),
                  Input('video_uploader', 'contents'),
                  State('video_uploader', 'filename'),
                  State('video_uploader', 'last_modified'),
                  State('session', 'data'))
    def update_output(content, name, date, session_data):
        if content is None:
            raise PreventUpdate

        if session_data is not None and 'upload_dir' in session_data:
            upload_dir = Path(session_data['upload_dir'])
        else:
            upload_dir = utils.random_upload_url(mkdir=True)
            session_data = {'upload_dir': str(upload_dir)}
        file = upload_dir / 'video.mp4'
        # TODO: video file conversion
        content_type, content_string = content.split(',')
        file.write_bytes(base64.b64decode(content_string))
        duration = int(utils.get_duration(file))
        slider = video_range_slider(duration)
        player = get_video_player(id='video_player', src=content)
        return player, slider, session_data


    @app.callback(Output('analyze_btn', 'disabled'),
                  Input('session', 'modified_timestamp'))
    def activate_btn(t):
        if t > 0: # TODO: does not seem to work...
            return False


    @app.callback(#Output('skel_graph', 'figure'),
                  Output('console', 'children'),
                  #Input('angle_graph', 'clickData'),
                  Input('angle_graph', 'clickData'))
    def show_frame(click_data):
        print(click_data)
        return json.dumps(click_data, indent=2)


    @app.callback(Output('pose_graph', 'figure'),
                  Output('angle_graph', 'figure'),
                  Input('analyze_btn', 'n_clicks'),
                  State('video_range', 'value'),
                  State('video_uploader', 'filename'),
                  State('session', 'data'),
                  )
    def analyze_clicked(nclicks, slider_value, videoname, session_data):
        if nclicks is None:
            raise PreventUpdate
        upload_dir = session_data['upload_dir']
        pose_3d, knee_angles = utils.run_estimation('video.mp4',
                                                    in_dir=upload_dir,
                                                    video_range=slider_value)
        skel_fig = figures.create_skeleton_fig(pose_3d)
        ang_fig = figures.create_angle_figure(knee_angles)
        return skel_fig, ang_fig

