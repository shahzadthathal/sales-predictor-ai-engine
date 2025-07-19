# api/app.py

from flask import Flask
from api.routes import v1
import os

def create_app():
    app = Flask(__name__)
    app.register_blueprint(v1, url_prefix="/api/v1")
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
