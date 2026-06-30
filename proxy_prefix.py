

class StripDuplicatedScriptName:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = environ.get('SCRIPT_NAME', '').rstrip('/')
        if prefix:
            path_info = environ.get('PATH_INFO', '')
            if path_info == prefix or path_info.startswith(prefix + '/'):
                environ['PATH_INFO'] = path_info[len(prefix):] or '/'
        return self.app(environ, start_response)
