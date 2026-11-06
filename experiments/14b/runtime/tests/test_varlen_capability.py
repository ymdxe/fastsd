import types
import unittest

try:
    from src.kvcache_varlen import supports_varlen_model
    _DEPS = True
except Exception:  # pragma: no cover - dependency gate
    _DEPS = False


@unittest.skipUnless(_DEPS, "Torch/Transformers are required for varlen capability tests")
class VarlenCapabilityTests(unittest.TestCase):
    def _model(self, model_type="llama"):
        attention = types.SimpleNamespace(**{
            name: object() for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        })
        layer = types.SimpleNamespace(
            self_attn=attention, input_layernorm=object(),
            post_attention_layernorm=object(), mlp=object(),
        )
        return types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=model_type, rope_scaling=None),
            model=types.SimpleNamespace(
                layers=[layer], embed_tokens=object(), norm=object(), rotary_emb=object(),
            ),
            lm_head=object(),
        )

    def test_incomplete_llama_is_rejected(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="llama"),
            model=types.SimpleNamespace(layers=[]),
        )
        self.assertFalse(supports_varlen_model(model))

    def test_qwen3_shape_is_accepted(self):
        model = self._model("qwen3")
        model.model.layers[0].self_attn.q_norm = object()
        model.model.layers[0].self_attn.k_norm = object()
        self.assertTrue(supports_varlen_model(model))

    def test_llama_without_qk_norm_is_accepted(self):
        self.assertTrue(supports_varlen_model(self._model()))

    def test_dynamic_rope_is_rejected(self):
        model = self._model()
        model.config.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
        self.assertFalse(supports_varlen_model(model))

    def test_missing_projection_is_rejected(self):
        model = self._model()
        del model.model.layers[0].self_attn.v_proj
        self.assertFalse(supports_varlen_model(model))


if __name__ == "__main__":
    unittest.main()
