"""FS-Det Model Handler."""
from fvcore.common.file_io import PathHandler, PathManager

import logging


class FsDetHandler(PathHandler):
    """
    Resolve anything that's in FsDet model zoo.
    """

    PREFIX = "fsdet://"
    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.URL_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(FsDetHandler())
