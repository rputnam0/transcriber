"""Greedy equal-length batching for the installed MLX Qwen and MOSS decoders.

Uses the maintainer's model/audio preparation, cache and projection operations.
No audio padding or cross-recording language context is introduced. Keep a
single-versus-batch parity check before using a new runtime/model combination.
"""


def greedy_batch(model, waves, *, family, prompt=None, max_tokens=1024):
    import mlx.core as mx

    if not waves or len({len(w) for w in waves}) != 1:
        raise ValueError("Batch requires nonempty, equal-length audio")
    embeddings = []
    for wave in waves:
        if family == "moss":
            _, embedded, _, _ = model._prepare_generation_inputs(wave, prompt)
        elif family == "qwen":
            feats, mask, count = model._preprocess_audio(wave)
            audio_features = model.get_audio_features(feats, mask)
            ids = model._build_prompt(count, "English", None)
            embedded = model._build_inputs_embeds(ids, audio_features)[0]
        else:
            raise ValueError(family)
        embeddings.append(embedded)
    x = mx.stack(embeddings)
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    if cache is None:
        from mlx_audio.lm.models.cache import KVCache

        cache = [KVCache() for _ in range(model.config.text_config.num_hidden_layers)]
    token_embedding = (
        model.model.language_model.embed_tokens if family == "moss" else model.model.embed_tokens
    )

    def step(embedded):
        hidden = model.model(inputs_embeds=embedded, cache=cache)[:, -1:, :]
        logits = (
            model.lm_head(hidden)
            if model.lm_head is not None
            else token_embedding.as_linear(hidden)
        )
        return mx.argmax(logits[:, -1, :], axis=-1)

    current = step(x)
    mx.async_eval(current)
    done = [False] * len(waves)
    tokens = [[] for _ in waves]
    eos = model._eos_token_ids()
    for _ in range(max_tokens):
        following = step(token_embedding(current[:, None]))
        mx.async_eval(following)
        for i, token in enumerate(current.tolist()):
            if not done[i]:
                if token in eos:
                    done[i] = True
                else:
                    tokens[i].append(token)
        if all(done):
            break
        current = following
    mx.eval(following)
    return [model._tokenizer.decode(t, skip_special_tokens=True).strip() for t in tokens], done
