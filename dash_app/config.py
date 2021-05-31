from flask.helpers import get_root_path

class DashConfig:
	ROOT_PATH = get_root_path(__name__)
	UPLOAD_ROOT = ROOT_PATH + '/uploads'
	ASSETS_ROOT = 'assets'
	DEMO_DATA = 'dash_app/assets'