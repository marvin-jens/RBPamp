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
    if formatter:
        handler.setFormatter(formatter)

    return handler

def getLogger(name, address="tcp://127.0.0.1:8888", formatter=None):
    logger = logging.getLogger(name)
    logger.addHandler(make_handler(address=address, formatter=formatter))
    return logger

def server_loop(address="tcp://*:8888", stream=sys.stdout):
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.bind(address)
    while True:
        lvl, msg = recv_socket.recv_multipart()
        stream.write(msg + '\n')
        stream.flush()

if __name__ == "__main__":
    FORMAT = '%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'
    formatter = logging.Formatter(FORMAT)

    if len(sys.argv) > 1:
        logger = getLogger('pushy', formatter=formatter)
        logger.warn(sys.argv[1])
    else:
        import argparse
        parser = argparse.ArgumentParser(description='Collect log messages from cska jobs on the cluster')
        # TODO: configure interface we're listening on, where to write, filters etc...
        server_loop()






