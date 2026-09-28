"""Stopping a running stage from outside (the web app's stop button).

``Cancelled`` derives from BaseException, like KeyboardInterrupt, so the
``except Exception`` blocks that keep a render going after one failed take
don't swallow it.
"""


class Cancelled(BaseException):
    """The user stopped the job."""
