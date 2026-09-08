"""Experimental text-only Gemma cache preserving native shared attention state."""

from prefix_cache import PrefixCache, clone


class GemmaPrefixCache(PrefixCache):
    def __init__(self, model, stop, logits):
        assert model.config.model_type == "gemma4_text"
        self.shared_updates = {}
        super().__init__(model, stop, logits)

    def forward(self, index, original, *args, **kwargs):
        if not self.active:
            return original(*args, **kwargs)
        assert kwargs.get("past_key_values") is None
        shared = kwargs["shared_kv_states"]
        before = dict(shared)
        output = super().forward(index, original, *args, **kwargs)
        if self.reuse:
            shared.update({key: clone(value) for key, value in self.shared_updates[index].items()})
        else:
            self.shared_updates[index] = {
                key: clone(value)
                for key, value in shared.items()
                if key not in before or value is not before[key]
            }
        return output

    def close(self):
        self.shared_updates.clear()
        super().close()
