# api/app.py

from flask import Flask, send_from_directory
from api.routes import v1
import os

def create_app():
    app = Flask(__name__, static_folder=os.path.abspath("public"))
    app.register_blueprint(v1, url_prefix="/api/v1")

    @app.route('/')
    def serve_index():
        return send_from_directory(app.static_folder, 'index.html')

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
