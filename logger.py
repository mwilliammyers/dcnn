import struct


class Logger(object):

	def __init__(self, file_path):
		self.fp = open(file_path, 'wb')

	def log(self, floats):
		data = struct.pack('f'*len(floats), *floats)
		self.fp.write(data)

	def close(self):
		self.fp.close()