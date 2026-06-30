import platform

_USE_PGSERVER = platform.machine() in ("x86_64", "amd64")


def using_pgserver():
    return _USE_PGSERVER


def start_embedded(data_dir):
    if _USE_PGSERVER:
        import database
        from linux import env

        with env.native_lib_path_restored():
            return database.start_embedded(data_dir)
    from linux import embedded_pg

    return embedded_pg.start(data_dir)


def ensure_embedded_running(data_dir):
    if _USE_PGSERVER:
        import database
        from linux import env

        with env.native_lib_path_restored():
            return database.ensure_embedded_running(data_dir)
    from linux import embedded_pg

    return embedded_pg.ensure_running(data_dir)


def stop_embedded():
    if _USE_PGSERVER:
        import database

        return database.stop_embedded()
    from linux import embedded_pg

    return embedded_pg.stop()
