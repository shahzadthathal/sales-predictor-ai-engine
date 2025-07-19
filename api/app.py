# api/app.py

from flask import Flask
from api.routes import v1

def create_app():
    app = Flask(__name__)
    app.register_blueprint(v1)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)