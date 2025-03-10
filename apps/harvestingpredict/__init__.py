from flask import Blueprint

blueprint = Blueprint(
    'harvestingpredict_blueprint',
    __name__,
    url_prefix='/harvesta-api/harvestingpredict'
)
