from flask import Flask
from flask.helpers import get_root_path

import dash
import dash_bootstrap_components as dbc

def create_server():
    server = Flask(__name__)
    #server.config.from_object(BaseConfig)

    register_dashapps(server)
    #register_extensions(server)
    #register_blueprints(server)

    return server


def register_dashapps(server):
    from dash_app.layout import layout
    from dash_app.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    app = dash.Dash(__name__,
                         server=server,
                         url_base_pathname='/',
                         assets_folder=get_root_path(__name__) + '/assets/',
                         meta_tags=[meta_viewport],
                         )

    with server.app_context():
        app.title = 'Gait Analyzer'
        app.layout = layout
        register_callbacks(app)