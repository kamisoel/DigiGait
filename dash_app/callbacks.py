#from dash.dependencies import Input, Output, State
from dash import no_update
from dash_extensions.enrich import Output, Input, State, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate

from dash_app.layout import video_range_slider, get_video_player
from dash_app import utils
from dash_app import figures

import base64
from pathlib import Path
from collections import defaultdict
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
        return json.dumps(click_data, indent=2)


    @app.callback(Output('pose_graph', 'figure'),
                  Output('angle_graph', 'figure'),
                  Output('gait_phase_graph', 'figure'),
                  Input('pose_data', 'data'),
                  State('option_boxes','value'),)
    def update_figures(data, options):
        pose_3d = data['pose']
        knee_angles = data['angles']
        gait_cycles = data['rcycles']
        avg_gait_phase = data['avg_phase']
        norm_data = data['norm_data']

        eye = utils.get_sagital_view(pose_3d)
        skel_fig = figures.create_skeleton_fig(pose_3d, eye=eye)
        if 'show_cycles' not in options:
            gait_cycles = []
        ang_fig = figures.create_angle_figure(knee_angles, gait_cycles)
        gait_phase_fig = figures.create_gait_phase_figure(
                            avg_gait_phase, norm_data)
        return skel_fig, ang_fig, gait_phase_fig


    @app.callback(ServersideOutput('pose_data', 'data'),
                  Output("error-toast", "is_open"), 
                  Trigger('analyze_btn', 'n_clicks'),
                  State('video_data', 'data'),
                  State('video_range', 'value'),
                  State('estimator_select', 'value'),
                  State('option_boxes','value'),
                  )
    def analyze_clicked(video_content, slider_value, pipeline, options):
        try:
            #upload_dir = session_data['upload_dir']
            video_path = utils.memory_file(video_content)
            ops = defaultdict(bool, {k: (k in options) for k in options})
            pose_3d, knee_angles, gait_cycles = utils.run_estimation(video_path, slider_value, pipeline, ops)
            avg_gait_phase = utils.avg_gait_phase(knee_angles, gait_cycles)

            norm_data = utils.get_norm_data()['Knee']
            return dict(pose=pose_3d, angles=knee_angles, rcycles=gait_cycles[0], lcycles=gait_cycles[1],
                        avg_phase=avg_gait_phase, norm_data=norm_data), no_update
        except Exception as e:
            return no_update, True

