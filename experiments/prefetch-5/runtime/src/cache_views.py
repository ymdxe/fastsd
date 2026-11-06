"""Build DynamicCache metadata without reading or copying in-flight storage."""

from transformers.cache_utils import DynamicCache


def dynamic_cache_from_views(legacy, *, buffers=None):
    # Newer Transformers DynamicLayer.update concatenates even the first
    # update. Build initialized empty metadata, then install the actual views.
    # Empty updates never read pinned CPU buffers or launch data copies.
    cache = DynamicCache.from_legacy_cache([(k[..., :0, :], v[..., :0, :]) for k, v in legacy])
    if hasattr(cache, "layers"):
        for layer, (key, value) in zip(cache.layers, legacy):
            layer.keys = key
            layer.values = value
    else:
        cache.key_cache = [k for k, _ in legacy]
        cache.value_cache = [v for _, v in legacy]
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = legacy[0][0].shape[-2] if legacy else 0
    if buffers is not None:
        cache._fastsd_buffers = buffers
        if reusable_cache_buffers(cache) is None:
            raise ValueError('Reusable KV buffers must own the supplied cache views')
    return cache


def reusable_cache_buffers(cache):
    """Reject stale/device-moved buffers without keeping a shadow GPU cache."""
    if cache is None:
        return None
    buffers = getattr(cache, '_fastsd_buffers', None)
    if buffers is None:
        return None
    legacy = cache.to_legacy_cache()
    valid = len(buffers) == len(legacy)
    if valid:
        for views, storage in zip(legacy, buffers):
            if len(views) != 2 or len(storage) != 2:
                valid = False
                break
            for view, buffer in zip(views, storage):
                if (view.device != buffer.device or view.dtype != buffer.dtype
                        or view.shape[:-2] != buffer.shape[:-2] or view.shape[-1] != buffer.shape[-1]
                        or view.shape[-2] > buffer.shape[-2]
                        or view.storage_offset() != buffer.storage_offset()
                        or view.stride() != buffer.stride()
                        or view.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()):
                    valid = False
                    break
            if not valid:
                break
    if not valid:
        del cache._fastsd_buffers
        return None
    return buffers
