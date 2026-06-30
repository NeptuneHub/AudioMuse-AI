import datetime


UTC_NOW_SQL = "NOW() AT TIME ZONE 'UTC'"

LOCAL_TZ_FMT = '%Y-%m-%d %H:%M:%S'


def to_local(dt):
    if dt is None or not hasattr(dt, 'astimezone'):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone()


def to_local_str(dt):
    dt = to_local(dt)
    if dt is None:
        return None
    if hasattr(dt, 'strftime'):
        return dt.strftime(LOCAL_TZ_FMT)
    return str(dt)
