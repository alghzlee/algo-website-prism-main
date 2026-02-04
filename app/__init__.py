from flask import Flask
from config import Config
from app.extensions import socketio
from pymongo import MongoClient
from flask_cors import CORS
import certifi

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Connect to MongoDB Atlas with SSL certificate
    client = MongoClient(app.config['MONGODB_URL'], tlsCAFile=certifi.where())
    app.db = client[app.config['DBNAME']]
    
    # Assets route for serving files from MongoDB GridFS
    from .routes.assets import assets_
    app.register_blueprint(assets_)
    
    from .routes.auth import auth_
    app.register_blueprint(auth_)
    
    from .routes.profile import profile_
    app.register_blueprint(profile_)
    
    from .routes.bed_selection import home_
    app.register_blueprint(home_)
    
    from .routes.similarity import similarity_
    app.register_blueprint(similarity_)
    
    from .routes.vital_prediction import prediction_
    app.register_blueprint(prediction_)
    
    from .routes.patient_treatment import treatments_
    app.register_blueprint(treatments_)
    
    from .routes.summary import summary_
    app.register_blueprint(summary_)
    
    from .routes.predict import predict_
    app.register_blueprint(predict_)
    
    socketio.init_app(app, cors_allowed_origins="*", engineio_logger=True, logger=True)

    from .websockets.patient_monitoring_socket import patientMonitoringSocketio
    app.register_blueprint(patientMonitoringSocketio)
    
    from .websockets.data_prediction_socket import dataPredictionSocketio
    app.register_blueprint(dataPredictionSocketio)
    
    from .websockets.treatment_recommendation_socket import treatmentRecommendationSocketio
    app.register_blueprint(treatmentRecommendationSocketio)
    
    # Simple health check endpoint (no model loading)
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'service': 'prism-api'}, 200
    
    # PRELOAD ML MODELS - Dengan preload_app=True, models di-load di master process
    # Ini menghindari loading ulang setiap kali worker restart (timeout issue)
    # Model akan di-share ke semua worker via fork (copy-on-write)
    print("[Startup] Preloading ML models (master process)...")
    
    try:
        from .routes.predict import get_model, get_physpol
        
        # Load models SYNCHRONOUSLY di master process
        print("[Startup] Loading SAC Ensemble (121.9 MB)...")
        get_model()      # SAC Ensemble (122MB)
        print("[Startup] Loading Physician Policy...")
        get_physpol()    # Physician policy
        
        print("[Startup] ✓ ML models preloaded - ready to fork workers!")
        
    except Exception as e:
        # Fatal: jika preload gagal, worker juga akan gagal
        print(f"[Startup] ✗ Model preload FAILED: {e}")
        raise
    
    return app