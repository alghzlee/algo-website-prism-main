from flask import Flask, jsonify
from config import Config
from app.extensions import socketio
from pymongo import MongoClient
from flask_cors import CORS
import certifi
import time
from datetime import datetime

# Global model status tracker
model_status = {
    'ensemble': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start', 'timestamp': None},
    'physician_policy': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start', 'timestamp': None},
    'autoencoder': {'status': 'pending', 'progress': 0, 'message': 'Waiting to start', 'timestamp': None},
    'startup_time': datetime.utcnow().isoformat(),
    'ready': False
}

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
    
    # Simple health check endpoint (Railway uses this)
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'service': 'prism-api'}, 200
    
    # Detailed model status check
    @app.route('/health/models')
    def model_health_check():
        """Check ML model download/loading status in real-time"""
        global model_status
        
        # Calculate overall progress
        total_progress = sum(m['progress'] for m in [model_status['ensemble'], 
                                                       model_status['physician_policy'],
                                                       model_status['autoencoder']])
        avg_progress = total_progress / 3
        
        # Determine overall status
        all_ready = all(m['status'] == 'ready' for m in [model_status['ensemble'], 
                                                           model_status['physician_policy'],
                                                           model_status['autoencoder']])
        any_error = any(m['status'] == 'error' for m in [model_status['ensemble'], 
                                                           model_status['physician_policy'],
                                                           model_status['autoencoder']])
        
        overall_status = 'ready' if all_ready else ('error' if any_error else 'loading')
        
        return jsonify({
            'status': overall_status,
            'ready': model_status['ready'],
            'progress': round(avg_progress, 1),
            'models': {
                'ensemble': model_status['ensemble'],
                'physician_policy': model_status['physician_policy'],
                'autoencoder': model_status['autoencoder']
            },
            'startup_time': model_status['startup_time'],
            'current_time': datetime.utcnow().isoformat()
        }), 200
    
    # PRELOAD ML MODELS - Optimized for Railway deployment with eventlet
    # Background preloading warms up worker & caches heavy libraries (torch, numpy)
    # Uses eventlet.spawn instead of threading to avoid conflicts
    import os
    
    def preload_ml_models():
        """
        Preload ML models in background with real-time status tracking:
        1. Non-blocking startup (Railway health check passes immediately)
        2. Download + load models with progress updates
        3. Update global model_status for monitoring
        """
        global model_status
        
        try:
            # Only in production to avoid blocking local dev
            if not os.environ.get('RAILWAY_ENVIRONMENT'):
                return
            
            import eventlet
            
            # Short delay to let Flask finish startup (non-blocking)
            eventlet.sleep(2)
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 🚀 Background model preloading started")
            
            # Update status: starting
            model_status['ensemble']['status'] = 'downloading'
            model_status['ensemble']['message'] = 'Starting download...'
            model_status['ensemble']['timestamp'] = datetime.utcnow().isoformat()
            
            from .routes.predict import get_model, get_physpol
            
            # Load SAC Ensemble (122MB - heaviest)
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 📦 Downloading Ensemble SAC model (122MB)...")
            model_status['ensemble']['progress'] = 10
            start_time = time.time()
            
            ensemble = get_model()
            
            elapsed = time.time() - start_time
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ✓ Ensemble loaded in {elapsed:.1f}s")
            model_status['ensemble']['status'] = 'ready'
            model_status['ensemble']['progress'] = 100
            model_status['ensemble']['message'] = f'Loaded in {elapsed:.1f}s'
            model_status['ensemble']['timestamp'] = datetime.utcnow().isoformat()
            
            # Load Physician Policy
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 📦 Loading physician policy...")
            model_status['physician_policy']['status'] = 'loading'
            model_status['physician_policy']['progress'] = 50
            
            physpol = get_physpol()
            
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ✓ Physician policy ready")
            model_status['physician_policy']['status'] = 'ready'
            model_status['physician_policy']['progress'] = 100
            model_status['physician_policy']['message'] = 'Ready'
            model_status['physician_policy']['timestamp'] = datetime.utcnow().isoformat()
            
            # AutoEncoder (included in ensemble)
            model_status['autoencoder']['status'] = 'ready'
            model_status['autoencoder']['progress'] = 100
            model_status['autoencoder']['message'] = 'Loaded with ensemble'
            model_status['autoencoder']['timestamp'] = datetime.utcnow().isoformat()
            
            # Mark as fully ready
            model_status['ready'] = True
            
            total_time = time.time() - start_time
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 🎉 All models ready! Total: {total_time:.1f}s")
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 🔍 Check status: /health/models")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ❌ Model preload failed: {error_msg}")
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Traceback:\n{traceback.format_exc()}")
            
            # Update error status
            model_status['ensemble']['status'] = 'error'
            model_status['ensemble']['message'] = error_msg[:100]
            model_status['physician_policy']['status'] = 'error'
            model_status['physician_policy']['message'] = 'Failed due to ensemble error'
            model_status['ready'] = False
            
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ⚠️  Models will lazy-load on first request")
    
    # Start preloading in eventlet greenthread (production only)
    import os
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        import eventlet
        eventlet.spawn(preload_ml_models)
    print("[Startup] Model preloading scheduled (background thread)")
    
    return app