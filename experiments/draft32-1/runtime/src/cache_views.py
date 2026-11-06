"""Build DynamicCache metadata without reading or copying in-flight storage."""

from transformers.cache_utils import DynamicCache


def dynamic_cache_from_views(legacy):
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
    return cache
