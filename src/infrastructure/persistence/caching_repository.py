"""
Generic caching wrapper for repositories.

Provides transparent caching layer to reduce database queries
without changing repository interfaces or UI code.
"""
from typing import TypeVar, Generic, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from threading import RLock

from src.utils.message import Log


T = TypeVar('T')


class CachedRepository(Generic[T]):
    """
    Generic caching wrapper for repositories.
    
    Wraps any repository and adds transparent caching with TTL (time-to-live).
    Cache is thread-safe and automatically expires stale entries.
    
    Example:
        block_repo = SQLiteBlockRepository(database)
        cached_repo = CachedRepository(block_repo, ttl_seconds=60)
        
        # First call: fetch from database
        block = cached_repo.get(project_id, block_id)
        
        # Second call (within 60s): return from cache
        block = cached_repo.get(project_id, block_id)  # Fast!
        
        # After 60s: cache expired, fetch from database again
        block = cached_repo.get(project_id, block_id)
    """
    
    def __init__(
        self,
        inner_repository: Any,
        ttl_seconds: int = 60,
        max_size: int = 1000
    ):
        """
        Initialize caching wrapper.
        
        Args:
            inner_repository: The repository to wrap
            ttl_seconds: Time-to-live for cache entries (default 60s)
            max_size: Maximum number of entries to cache (default 1000)
        """
        self.inner = inner_repository
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        # Cache storage: {cache_key: (value, expiry_time)}
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        
        # Thread-safe lock
        self._lock = RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def get(self, *args, **kwargs):
        """
        Get entity with caching.
        
        Delegates to inner repository's get() method with caching.
        """
        # Generate cache key from arguments
        cache_key = self._make_cache_key('get', args, kwargs)
        
        with self._lock:
            # Check cache
            if cache_key in self._cache:
                value, expiry = self._cache[cache_key]
                
                # Check if expired
                if datetime.now() < expiry:
                    self._hits += 1
                    # Calculate hit rate before logging to avoid recursion
                    try:
                        hit_rate = self.hit_rate
                        Log.debug(f"Cache HIT: {cache_key} (hit rate: {hit_rate:.1%})")
                    except:
                        # Prevent recursion in logging
                        pass
                    return value
                else:
                    # Expired, remove from cache
                    del self._cache[cache_key]
            
            # Cache miss - fetch from inner repository
            self._misses += 1
            try:
                hit_rate = self.hit_rate
                Log.debug(f"Cache MISS: {cache_key} (hit rate: {hit_rate:.1%})")
            except:
                # Prevent recursion in logging
                pass
            
            value = self.inner.get(*args, **kwargs)
            
            # Store in cache
            if value is not None:
                self._put_cache(cache_key, value)
            
            return value
    
    def invalidate(self, *args, **kwargs):
        """
        Invalidate cache entry for given arguments.
        
        Call this after updating an entity to ensure cache stays fresh.
        """
        cache_key = self._make_cache_key('get', args, kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                try:
                    Log.debug(f"Cache invalidated: {cache_key}")
                except:
                    # Prevent recursion in logging
                    pass
    
    def clear_cache(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            try:
                Log.info("Cache cleared")
            except:
                # Prevent recursion in logging
                pass
    
    def _put_cache(self, key: str, value: Any):
        """
        Put value in cache with TTL.
        
        Implements LRU eviction if cache is full.
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            # Evict oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            try:
                Log.debug(f"Cache evicted: {oldest_key} (cache full)")
            except:
                # Prevent recursion in logging
                pass
        
        # Store with expiry time
        expiry = datetime.now() + timedelta(seconds=self.ttl_seconds)
        self._cache[key] = (value, expiry)
    
    def _make_cache_key(self, method: str, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key from method name and arguments.
        
        Args:
            method: Method name (e.g., 'get')
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Convert args and kwargs to stable string representation
        args_str = ':'.join(str(arg) for arg in args)
        kwargs_str = ':'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        if kwargs_str:
            return f"{method}:{args_str}:{kwargs_str}"
        else:
            return f"{method}:{args_str}"
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'cache_size': self.cache_size,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }
    
    def __getattr__(self, name: str):
        """
        Delegate all other methods to inner repository.
        
        This allows the wrapper to be transparent - any method not
        explicitly defined here will be forwarded to the wrapped repository.
        """
        return getattr(self.inner, name)


class CachedBlockRepository(CachedRepository):
    """
    Specialized caching wrapper for BlockRepository.
    
    Overrides update/create/delete to invalidate cache automatically.
    """
    
    def create(self, block):
        """Create block and cache it."""
        result = self.inner.create(block)
        # No need to cache create - it's a new entity
        return result
    
    def update(self, block):
        """Update block and invalidate cache."""
        result = self.inner.update(block)
        # Invalidate cache for this block
        self.invalidate(block.project_id, block.id)
        return result
    
    def delete(self, project_id: str, block_id: str):
        """Delete block and invalidate cache."""
        result = self.inner.delete(project_id, block_id)
        # Invalidate cache for this block
        self.invalidate(project_id, block_id)
        return result


class CachedConnectionRepository(CachedRepository):
    """Specialized caching wrapper for ConnectionRepository."""
    
    def create(self, connection):
        """Create connection."""
        return self.inner.create(connection)
    
    def delete(self, connection_id: str):
        """Delete connection and clear cache (connections affect graph structure)."""
        result = self.inner.delete(connection_id)
        # Clear entire cache since connections affect list queries
        self.clear_cache()
        return result


class CachedProjectRepository(CachedRepository):
    """Specialized caching wrapper for ProjectRepository."""
    
    def update(self, project):
        """Update project and invalidate cache."""
        result = self.inner.update(project)
        self.invalidate(project.id)
        return result


