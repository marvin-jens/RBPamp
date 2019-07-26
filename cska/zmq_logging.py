# -*- coding: future_fstrings -*-
from __future__ import print_function
__license__ = "MIT"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import sys
import re
import logging
import zmq
from zmq.log.handlers import PUBHandler
from collections import defaultdict
import numpy as np


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


class RBPStates(object):
    def __init__(self):
        import cska
        self.rbps = sorted(cska.dominguez_rbps)
        self.corrs_by_rbp = defaultdict(dict)
        self.t_by_rbp = defaultdict(dict)
        self.symbols = {
            'std.8' : 'O',
            'std.8.1' : '1',
            'std.8.xsrbp' : 'x',
            'std.8.linocc' : 'l',
            'std.8.dumb' : 'd'
        }

    def parse(self, msg):
        parts = re.split('\s+', msg)
        if "max_corr" in msg:
            rbp = parts[6]
            run = parts[7]
            if not run in self.symbols:
                return

            m = re.search(r'max_corr=(\S+)', msg)
            corr = float(m.groups()[0])

            self.corrs_by_rbp[rbp][run] = corr
            
            m = re.search(r't=(\S+)', msg)
            t = int(m.groups()[0])
            self.t_by_rbp[rbp][run] = t
    
    def __str__(self):
        rows = []
        
        for rbp in self.rbps:
            if not rbp in self.corrs_by_rbp:
                continue
            
            corrs = [" ",] * 51
            for run, c in sorted(self.corrs_by_rbp[rbp].items()):
                x = int(c * 50)
                corrs[x] = self.symbols[run]
            
            cstr = "".join(corrs)

            times = [" ",] * 51
            for run, t in sorted(self.t_by_rbp[rbp].items()):
                x = min(t/4, 50)
                times[x] = self.symbols[run]

            tstr = "".join(times)

            runs = sorted(self.symbols.keys())
            ctupstr = ",".join(["{}={:.3f}".format(self.symbols[run], np.round(self.corrs_by_rbp[rbp].get(run, np.NaN), 3)) for run in runs])
            ttupstr = ",".join(["{}={:03d}".format(self.symbols[run], self.t_by_rbp[rbp].get(run, -1)) for run in runs])
        
            rows.append("{rbp:20s}| {cstr} | {tstr} | corrs = {ctupstr} times = {ttupstr}".format(**locals()))
        return "\n".join(rows)

def opt_stats_loop(address="tcp://*:8888", stream=sys.stdout):
    import os
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.bind(address)
    
    states = RBPStates()

    while True:
        rec = recv_socket.recv_multipart()
        if len(rec) != 2:
            stream.write('received malformed message "{}" \n'.format(rec))
        else:
            lvl, msg = rec
            states.parse(msg)
            os.system('clear')
            print(states)
            # stream.write(msg + '\n')

        stream.flush()


class StateTrackerData(object):
    def __init__(self, run, rbps):
        self.rbp_states = {}
        self.run = run
        self.rbps = rbps
    
    def parse(self, msg):
        if not "main.StateTracker" in msg:
            return
        
        parts = re.split('\t', msg)
        lvl = parts[1]
        if lvl != 'INFO':
            return

        git = parts[3]
        rbp = parts[4]
        run = parts[5]
        stage = parts[8]
        ts = parts[9]
        status = parts[10]

        if run == self.run:
            self.rbp_states[rbp] = (stage, status, ts, git)
    
    def __str__(self):
        header = "\t".join(["RBP", 'stage', 'git commit', 'time', 'status'])

        buf = [header]
        for rbp in self.rbps:
            if rbp in self.rbp_states:
                (stage, status, ts, git) = self.rbp_states[rbp]
                buf.append(f"{rbp:20s}\t{stage:20s}\t{git}\t{ts}\t{status}")

        return '\n'.join(buf)
     
    def load_logs(self, pattern):
        from glob import glob
        for flog in glob(pattern):
            for line in file(flog):
                self.parse(line)


def state_tracker_loop(address="tcp://*:8888", stream=sys.stdout):
    import os
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.bind(address)
    
    import cska
    states = StateTrackerData("z4t75p01k99fix", cska.dominguez_rbps)
    states.load_logs('/home/mjens/engaging/RBNS/*/cska/z4t75p01k99fix/run.log')

    while True:
        os.system('clear')
        print(states)
        rec = recv_socket.recv_multipart()
        if len(rec) != 2:
            stream.write('received malformed message "{}" \n'.format(rec))
        else:
            lvl, msg = rec
            states.parse(msg)
            # stream.write(msg + '\n')

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
        # server_loop()
        # opt_stats_loop()
        state_tracker_loop()






