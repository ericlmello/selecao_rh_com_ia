import os
import multiprocessing

# Configurações básicas
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
workers = 1  # Apenas 1 worker para economizar memória
worker_class = "sync"
worker_connections = 100

# Configurações de memória
max_requests = 100  # Reinicia worker após N requests
max_requests_jitter = 10
preload_app = True  # Pré-carrega app para economizar memória

# Timeouts
timeout = 60
keepalive = 30
worker_tmp_dir = "/dev/shm"  # Usa RAM para temp files

# Configurações de log
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Configurações de processo
user = None
group = None
daemon = False
pidfile = None
umask = 0
tmp_upload_dir = "/tmp"

# Configurações de SSL (se necessário)
keyfile = None
certfile = None

# Configurações específicas para economizar memória
def when_ready(server):
    server.log.info("Servidor pronto - memória otimizada")

def worker_int(worker):
    worker.log.info("Worker interrompido")

def pre_fork(server, worker):
    server.log.info("Worker iniciado")

def post_fork(server, worker):
    server.log.info("Worker fork completo")
    
def worker_abort(worker):
    worker.log.info("Worker abortado")

# Configuração de limite de memória
def on_starting(server):
    import resource
    # Limita memória virtual para 512MB
    resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    server.log.info("Limite de memória definido para 512MB")
