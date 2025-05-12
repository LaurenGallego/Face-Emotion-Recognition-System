from flask import Flask 
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager


db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'AISC'
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{path.abspath('website')}/{DB_NAME}"  
    db.init_app(app)

    from .views import views
    from .auth import auth
    from .video import video

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(video, url_prefix='/')

    from .models import User, Report

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login' 
    login_manager.init_app(app) 

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    

    return app


def create_database(app):
    # Specify the correct path for the database inside 'website'
    db_path = path.join('website', DB_NAME)
    print(f"Database path: {db_path}")  # Debugging line to check the path
    with app.app_context():
        if not path.exists(db_path):  # Check in the 'website' folder
            db.create_all()
            print('Created Database!')
        else:
            print(f'Database already exists at {db_path}')