import threading

class CurrentUserMiddleware:
    _thread_local = threading.local()

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        CurrentUserMiddleware._thread_local.user = request.user
        response = self.get_response(request)
        return response

    @staticmethod
    def get_current_user():
        return getattr(CurrentUserMiddleware._thread_local, 'user', None)
