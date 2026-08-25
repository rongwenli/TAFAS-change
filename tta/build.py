def build_adapter(cfg, model, norm_module=None):
    method = cfg.TTA.METHOD.upper()

    if method == 'TAFAS':
        from tta.tafas import build_adapter as build_tafas_adapter

        return build_tafas_adapter(cfg, model, norm_module)
    if method in ('CLOSED_FORM', 'PDCF'):
        from tta.closed_form.adapter import PredictionDerivedClosedFormAdapter

        return PredictionDerivedClosedFormAdapter(cfg, model, norm_module)

    raise ValueError(
        f"Unsupported TTA method: {cfg.TTA.METHOD}. "
        "Expected one of: TAFAS, CLOSED_FORM."
    )
