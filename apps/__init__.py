from flask import Flask
from importlib import import_module

def register_blueprints(app):
    modules = ['harvestingpredict']
    for module_name in modules:
        module = import_module(f'apps.{module_name}.routes')
        app.register_blueprint(module.blueprint)

def create_app(config_class='config.Config'):
    app = Flask(__name__)
    app.config.from_object(config_class)
    register_blueprints(app)
    return app
