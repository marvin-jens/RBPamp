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
    debug_caching = False # set to True to get A LOT of debug output from the caching framework
    
    # change any of the below, on instance or class level, to tune behaviour 
    # of the caching framework
    _do_not_cache = False
    _do_not_pickle = False
    _do_not_unpickle = False
    
    def __init__(self):
        self._cache_names = []
        self.logger = logging.getLogger('CachedBase')
        #self._do_not_cache = True # DEBUG!!

    @property
    def cache_key(self):
        """
        This needs to be overridden by each subclass, unless class attributes
        really do not influence the identity of the cached results.
        """
        return self.__class__.__name__
    
    def cache_preload(self, cache_name, value, key="."):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )

        getattr(self, cache_name)[key] = value
        if self.debug_caching:
            self.logger.debug("cache_preload {0} '{1}' to {2}".format(cache_name, key, value) )
    
    def cache_flush(self, cache_names = []):
        if not cache_names:
            cache_names = self._cache_names
        self.logger.debug("{0} flushing caches '{1}'".format(self.cache_key, cache_names) )
        for cache_name in cache_names:
            setattr(self, cache_name, dict() )

    def cache_debug(self):
        for name in self._cache_names:
            print ">>>", self.cache_key, name
            for k,v in sorted(getattr(self, name).items()):
                print "  '{0}' : '{1}'".format(k,v)
    
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
    
    def cached_func(self, *argc, **kwargs):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )
            self._cache_names.append(cache_name)

        if self.debug_caching:
            self.logger.debug("cached function {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )
                
        cache = getattr(self, cache_name)
        
        def to_str(x):
            if type(x) == np.ndarray:
                return "array_{0}".format(x.shape)
            else:
                return str(x)

        argc_key = "_".join([to_str(a) for a in argc])
        kw_key = "__".join(["{0}={1}".format(k,v) for k,v in sorted(kwargs.items()) ])
        key = argc_key + "." + kw_key
        
        if not key in cache:
            if self.debug_caching:
                self.logger.debug("{0} cache-miss '{1}'".format(cache_name, key) )
                #self.cache_debug()

            if getattr(self, '_do_not_cache', False):
                if self.debug_caching:
                    self.logger.debug("! NOT CACHING: calling {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )

                # override caching, but allow pre-loading!
                return func(self, *argc)
            else:
                if self.debug_caching:
                    self.logger.debug("! calling {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )
                cache[key] = func(self, *argc, **kwargs)
        else:
            if self.debug_caching:
                self.logger.debug("{0} cache-hit '{1}'".format(cache_name, key) )
            
        return cache[key]
    
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
        
        def to_str(x):
            if type(x) == np.ndarray:
                return "array_{0}".format(x.shape)
            else:
                return str(x)
        
        res = None
        new = False

        inst_key = self.cache_key
        argc_key = "_".join([to_str(a) for a in argc])
        kw_key = "__".join(["{0}={1}".format(k,v) for k,v in sorted(kwargs.items()) ])
        
        path = self.pkl_path
        pkl_name = "{inst_key}.{func.__name__}.{argc_key}.{kw_key}.pkl".format(**locals() )

        # get the result from call or un-pickle
        if getattr(self, '_do_not_unpickle', False):
            res = func(self, *argc, **kwargs)
            new = True

        elif os.path.exists(os.path.join(path,pkl_name)):
            self.logger.debug("un-pickling '{0}'".format(pkl_name) )
            res = pickle.load(file(os.path.join(path,pkl_name),'rb'))
            new = False
            
        else:
            res = func(self, *argc, **kwargs)
            new = True

        # store the result, if new and not disabled
        if new and len(res) and (not getattr(self, '_do_not_pickle', False)):
            self.logger.debug("storing pickle of '{0}'".format(pkl_name) )
            try:
                os.makedirs(path)
            except OSError:
                # already exists
                pass
            pickle.dump(res, file(os.path.join(path,pkl_name),'wb'), protocol=-1)
        
        return res
    
    pickled_func.__name__ = func.__name__
    return pickled_func
