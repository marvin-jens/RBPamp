__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import logging
import numpy as np
import cPickle as pickle

class CachedBase(object):
    """
    Base class for anything that wants to use transparent caching and/or 
    pickling by use of the @cached or @pickled decorators. Adds the 
    minimum hooks required to make this work.
    """

    pkl_path = "./.pkl/"
    debug_caching = True # set to True to get a lot of debug output from the caching framework

    def __init__(self):
        self._cache_names = []
        self.logger = logging.getLogger('CachedBase')
        self._do_not_cache = True # DEBUG!!

    def cache_key(self):
        """
        This needs to be overridden by each subclass, unless class attributes
        really do not influence the identity of the cached results.
        """
        return self.__class__.__name__
    
    def cache_preload(self, cache_name, value, argc=""):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )

        getattr(self, cache_name)[argc] = value
        if self.debug_caching:
            self.logger.debug("cache_preload {0} '{1}' to {2}".format(cache_name, argc, value) )
    
    def cache_flush(self, cache_names = []):
        if not cache_names:
            cache_names = self._cache_names
        
        for cache_name in cache_names:
            setattr(self, cache_name, dict() )

        
def cached(func):
    """
    Decorator for class methods that keeps the results of the first call and 
    returns the cached result for subsequent calls. Works by adding a 
    "__cached_<func_name>" dictionary to the decorated method's class instance.
    """
    # TODO: 
    # * mechanism to pre-populate cache (for subsample seqm)
    # * clean up into baseclass (or meta class?) of its own
    # * better handling of arrays as keys: Use hash function on data rather than shape.
    
    cache_name = "__cached_{name}".format(name=func.__name__)
    
    def cached_func(self, *argc):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )
            self._cache_names.append(cache_name)
            if self.debug_caching:
                self.logger.debug("accessed {0} for first time".format(cache_name) )
        
        cache = getattr(self, cache_name)
        
        def to_str(x):
            if type(x) == np.ndarray:
                return "array_{0}".format(x.shape)
            else:
                return str(x)

        argc_key = "_".join([to_str(a) for a in argc])

        if not argc_key in cache:
            if self.debug_caching:
                self.logger.debug("{0} cache-miss '{1}'".format(cache_name, argc_key) )

            if getattr(self, '_do_not_cache', False):
                # override caching, but allow pre-loading!
                return func(self, *argc)
            else:
                cache[argc_key] = func(self, *argc)
        else:
            if self.debug_caching:
                self.logger.debug("{0} cache-hit '{1}'".format(cache_name, argc_key) )
            
        return cache[argc_key]
    
    cached_func.__name__ = func.__name__
    return cached_func
  
def pickled(func):
    """
    Decorator for class methods that returns an un-pickled result if it exists. 
    Otherwise, stores the result of the call in a pickle file. Requires that the 
    class has an out_path attribute and a pickle_key method that returns a distinct 
    key for all the parameters that influence the results, ensuring that the correct
    object is unpickled.
    """
    
    def pickled_func(self, *argc, **kwargs):
        
        if getattr(self, '_do_not_cache', False):
            return func(self, *argc)
        
        def to_str(x):
            if type(x) == np.ndarray:
                return "array_{0}".format(x.shape)
            else:
                return str(x)

        inst_key = self.cache_key()
        argc_key = "_".join([to_str(a) for a in argc])
        kw_key = "__".join(["{0}={1}".format(k,v) for k,v in sorted(kwargs.items()) ])
        
        path = self.pkl_path
        pkl_name = "{inst_key}.{func.__name__}.{argc_key}.{kw_key}.pkl".format(**locals() )
        if not os.path.exists(os.path.join(path,pkl_name)):
            print "pickled: calling {0} with {1}".format(func.__name__, argc)
            res = func(self, *argc, **kwargs)
            try:
                os.makedirs(path)
            except OSError:
                # already exists
                pass
            if self.debug_caching:
                self.logger.debug("storing pickle of '{0}'".format(pkl_name) )
            pickle.dump(res, file(os.path.join(path,pkl_name),'wb'), protocol=-1)
        else:
            if self.debug_caching:
                self.logger.debug("un-pickling '{0}'".format(pkl_name) )
            res = pickle.load(file(os.path.join(path,pkl_name),'rb'))
        
        return res
    
    pickled_func.__name__ = func.__name__
    return pickled_func
