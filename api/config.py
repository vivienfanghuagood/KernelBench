"""
Configuration for KernelBench API
"""
import os

class APIConfig:
    """Configuration for the API service"""
    
    # Multiprocessing settings
    MAX_WORKERS = int(os.getenv("KERNELBENCH_MAX_WORKERS", "4"))
    WORKER_TIMEOUT = int(os.getenv("KERNELBENCH_WORKER_TIMEOUT", "300"))  # 5 minutes
    
    # Core dump settings (Linux only)
    CORE_DUMP_MAX_SIZE_MB = int(os.getenv("KERNELBENCH_CORE_DUMP_SIZE", "100"))
    
    # Process cleanup settings
    CLEANUP_INTERVAL = int(os.getenv("KERNELBENCH_CLEANUP_INTERVAL", "60"))  # seconds
    PROCESS_TERM_TIMEOUT = int(os.getenv("KERNELBENCH_TERM_TIMEOUT", "5"))  # seconds to wait before SIGKILL
    
    # Database settings
    DB_PATH = os.getenv("KERNELBENCH_DB_PATH", "kernelbench_api.db")
    
    # Generation settings
    DEFAULT_MAX_TOKENS = int(os.getenv("KERNELBENCH_DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE = float(os.getenv("KERNELBENCH_DEFAULT_TEMPERATURE", "0.0"))
    DEFAULT_NUM_CORRECT_TRIALS = int(os.getenv("KERNELBENCH_NUM_CORRECT_TRIALS", "3"))
    DEFAULT_NUM_PERF_TRIALS = int(os.getenv("KERNELBENCH_NUM_PERF_TRIALS", "50"))
