import os

# 基础配置
SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-key-123"

# 文件上传
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
