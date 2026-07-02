from app import app, load_model

# โหลด model ตอน gunicorn import startup module
load_model()

# expose app สำหรับ gunicorn
__all__ = ['app']
