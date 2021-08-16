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


    @app.callback(Output('animator', 'disabled'),
                  Input('autoscroll', 'checked'))
    def toggle_autoscroll(c):
        return not c


    @app.callback(Output('advanced_options', 'is_open'),
                  Output('options_btn', 'children'),
                  Trigger('options_btn', 'n_clicks'),
                  State('advanced_options', 'is_open'),
                  )
    def toggle_advanced_options(is_open):
        if is_open:
            return False, "► Advanced options"
        else:
            return True, "▼ Advanced options"


    app.clientside_callback(
    """
    function(n, angle_fig) {
        // check if a video is analyzed at the moment (don't update then)
        is_loading = document.querySelector('#pose_card .card-body div').style.visibility == 'hidden'
        if(is_loading)
            return window.dash_clientside.no_update;
        // get current frame by reading the counter next to the slider
        frame = parseInt(document.querySelector('#pose_graph svg text').textContent.split(':')[1]);
        // don't update if it's the same frame
        if(angle_fig.layout.shapes[0].x1 == frame) 
            return window.dash_clientside.no_update;
        // update position marker and move view if necessary
        new_angle_fig = Object.assign({}, angle_fig);
        new_axis_pos = Math.max(0, frame-150)
        if (new_angle_fig.layout.xaxis.range[0] < new_axis_pos || new_axis_pos == 0)
            new_angle_fig.layout.xaxis.range = [new_axis_pos, new_axis_pos+300]
        pos_marker = new_angle_fig.layout.shapes[0]
        pos_marker.x0 = Math.max(0,frame-5);
        pos_marker.x1 = frame+5;

        return new_angle_fig;
    }
    """,
    Output('angle_graph', 'figure'),
    Input('animator', 'n_intervals'),
    State('angle_graph','figure'),
    )


    @app.callback(Output('pose_graph', 'figure'),
                  Output('angle_graph', 'figure'),
                  Output('gait_phase_graph', 'figure'),
                  Input('pose_data', 'data'),
                  State('option_boxes','value'),)
    def update_figures(data, options):
        pose_3d = data['pose']
        knee_angles = data['angles']
        events = data['events']
        rhs = events[0]
        
        norm_data = utils.get_norm_data('overground')['KneeZ']
        avg_gait_phase = utils.avg_gait_phase(knee_angles, events)

        eye = utils.get_sagital_view(pose_3d)
        skel_fig = figures.create_skeleton_fig(pose_3d, eye=eye)
        if 'show_cycles' not in options:
            rhs = []
        ang_fig = figures.create_angle_figure(knee_angles, rhs)
        gait_phase_fig = figures.create_gait_phase_figure(
                            avg_gait_phase, norm_data)
        return skel_fig, ang_fig, gait_phase_fig


    @app.callback(ServersideOutput('pose_data', 'data'),
                  Output("error-toast", "is_open"), 
                  Trigger('analyze_btn', 'n_clicks'),
                  State('video_data', 'data'),
                  State('video_range', 'value'),
                  State('estimator_select', 'value'),
                  State('event_detection_select', 'value'),
                  State('option_boxes','value'),
                  )
    def analyze_clicked(video_content, slider_value, pipeline, detection, options):
        try:
            #upload_dir = session_data['upload_dir']
            video_path = utils.memory_file(video_content)
            ops = defaultdict(bool, {k: (k in options) for k in options})
            pose_3d, knee_angles, events = utils.run_estimation(video_path, slider_value, pipeline, detection, ops)

            return dict(pose=pose_3d, angles=knee_angles, events=events), False
        except Exception as e:
            raise
            return no_update, True

