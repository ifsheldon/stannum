from taichi import __version__ as __ti_version

is_legacy_taichi = __ti_version < (0, 7, 26)


def check_field_needs_grad(field, needs_grad):
    if needs_grad is None:
        if is_legacy_taichi:
            raise Exception(
                f"You are using legacy Taichi (v{__ti_version[0]}.{__ti_version[1]}.{__ti_version[2]} < v0.7.26), "
                f"you need to specify needs_grad yourself when registering a field")
        else:
            return field.snode.needs_grad
    return needs_grad
