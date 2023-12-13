import logging
import os
import nbdev

def set_log_level (logger, log_level):
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)

def cd_root ():
    config=nbdev.config.get_config()
    os.chdir(config.config_path)