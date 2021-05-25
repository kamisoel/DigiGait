import os
from flask import Flask
from flask.helpers import get_root_path

import dash
from dash_extensions.enrich import DashProxy, FileSystemStore
from dash_extensions.enrich import MultiplexerTransform, ServersideOutputTransform, TriggerTransform

#from dash_app.extensions import cache

def create_server():
    server = Flask(__name__)
    #server.config.from_object(BaseConfig)

    #register_extensions(server)
    register_dashapps(server)
    #register_blueprints(server)

    return server


def register_dashapps(server):
    from dash_app.layout import layout
    from dash_app.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    #if os.environ[FLASK_ENV] == 'development'
    enrich_transforms = [TriggerTransform(),  MultiplexerTransform(),
                         ServersideOutputTransform(FileSystemStore('cache'))]

    app = DashProxy(server=server,
                    url_base_pathname='/',
                    assets_folder=get_root_path(__name__) + '/assets/',
                    meta_tags=[meta_viewport],
                    transforms=enrich_transforms,
                    prevent_initial_callbacks=True,)

    with server.app_context():
        app.title = 'Gait Analyzer'
        app.layout = layout
        register_callbacks(app)

def register_extensions(server):
    #from flask_cacheify import init_cacheify
    if server.debug:
        cache.init_app(server, config={'CACHE_TYPE': 'FileSystemCache', 
                                        'CACHE_DEFAULT_TIMEOUT': 3600, 
                                        'CACHE_DIR': 'cache',
                                        'CACHE_THRESHOLD': 10})
    else:
        cache.init_app(server, config={'CACHE_TYPE': 'RedisCache', 
                                       'CACHE_DEFAULT_TIMEOUT': 3600})


