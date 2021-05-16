#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logging.handlers import BaseRotatingHandler
import time
import os
import codecs

class MultiProcessSafeDailyRotatingFileHandler(BaseRotatingHandler):
    """Similar with `logging.TimedRotatingFileHandler`, while this one is
    - Multi process safe
    - Rotate at midnight only
    - Utc not supported
    """
    def __init__(self, filename, encoding=None, delay=False, utc=False, **kwargs):
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.baseFilename = filename
        self.currentFileName = self._compute_fn()
        BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)

    def shouldRollover(self, record):
        if self.currentFileName != self._compute_fn():
            return True
        return False

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.currentFileName = self._compute_fn()

    def _compute_fn(self):
        return self.baseFilename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        if self.encoding is None:
            stream = open(self.currentFileName, self.mode)
        else:
            stream = codecs.open(self.currentFileName, self.mode, self.encoding)
            # simulate file name structure of `logging.TimedRotatingFileHandler`
            if os.path.exists(self.baseFilename):
                try:
                    os.remove(self.baseFilename)
                except OSError:
                    pass
            try:
                # os.symlink(source, link_name)
                # Create a symbolic link pointing to source named link_name.
                # Availability: Unix.
                os.symlink(self.currentFileName, self.baseFilename)
            except OSError:
                pass
            return stream