import zmq
import logging
import sys
from zmq.log.handlers import PUBHandler


class PUSHHandler(PUBHandler):
    def format(self, record):
        """
        Restores proper formatter usage of logging.Handler 
        that was arbitrarily messed up by zmq PUBHandler.
        """
        if self.formatter:
            fmt = self.formatter
        else:
            fmt = logging._defaultFormatter
        
        return fmt.format(record)

def make_handler(address="tcp://127.0.0.1:8888", formatter=None):
    context = zmq.Context()
    log_socket = context.socket(zmq.PUSH)
    log_socket.connect(address)
    handler = PUSHHandler(log_socket)
    handler.setLevel(logging.DEBUG)
    if formatter:
        handler.setFormatter(formatter)

    return handler

class LoggerFactory(object):
    def __init__(self, address="", format_str='%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'):
        self.format_str = format_str
        self.address = address

    def getLogger(self, name):
        logger = logging.getLogger(name)
        formatter = logging.Formatter(self.format_str)
        
        if self.address:
            logger.addHandler(make_handler(address=self.address, formatter=formatter))
        
        return logger

def getLogger(name, address="tcp://127.0.0.1:8888", formatter=None):
    logger = logging.getLogger(name)
    logger.addHandler(make_handler(address=address, formatter=formatter))
    return logger

def server_loop(address="tcp://*:8888", stream=sys.stdout):
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.bind(address)
    while True:
        rec = recv_socket.recv_multipart()
        if len(rec) != 2:
            stream.write('received malformed message "{}" \n'.format(rec))
        else:
            lvl, msg = rec
            stream.write(msg + '\n')

        stream.flush()

if __name__ == "__main__":
    FORMAT = '%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'
    formatter = logging.Formatter(FORMAT)

    if len(sys.argv) > 1:
        logger = getLogger('', formatter=formatter)
        logger.setLevel(logging.DEBUG)
        l2 = logging.getLogger('meep')
        # l2.setLevel(logging.DEBUG)
        logger.warn(sys.argv[1])
        logger.debug(sys.argv[1])
        l2.debug('blup')
    else:
        import argparse
        parser = argparse.ArgumentParser(description='Collect log messages from cska jobs on the cluster')
        # TODO: configure interface we're listening on, where to write, filters etc...
        server_loop()






