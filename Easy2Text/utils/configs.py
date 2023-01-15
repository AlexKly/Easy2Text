from pathlib import Path


class Configurations:
    # Directories and files paths:
    cur_dir = Path(__file__).parent.parent.resolve()
    dir_models = cur_dir/'models'
    # Logging:
    format_log = '%(asctime)s - %(message)s'
