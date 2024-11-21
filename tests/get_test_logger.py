import os

import logging 

def get_test_logger(name, level=logging.INFO):
    log_file = 'tests.log'
    logger = logging.getLogger(name)
    
    filehandler = logging.FileHandler('tests.log')
    formatter = logging.Formatter("%(name)s [%(levelname)s] : %(message)s")
    filehandler.setFormatter(formatter)
    
    logger.addHandler(filehandler)  
    logger.setLevel(level)

    return logger

