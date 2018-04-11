import struct
import os


class Logger(object):
    def __init__(self, file_path):
        base = '/'.join(file_path.split('/')[:-1])
        if base and not os.path.isdir(base):
            os.makedirs(base)
        self.fp = open(file_path, 'wb')

    def log(self, floats):
        data = struct.pack('f' * len(floats), *floats)
        self.fp.write(data)

    def close(self):
        self.fp.close()
