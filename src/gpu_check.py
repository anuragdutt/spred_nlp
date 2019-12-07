from tensorflow.python.client import device_lib
import tensorflow as tf


def get_available_devices():
	local_devices_protos = device_lib.list_local_devices()
	return [x.name for x in local_devices_protos]

if __name__ == "__main__":
	print(get_available_devices())
	print(tf.__version__)