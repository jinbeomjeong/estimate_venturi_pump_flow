import struct
import onnxruntime as ort
import numpy as np

from utils.udp_lib import UdpServer


model = ort.InferenceSession('models/reg_model.onnx')


udp_handle = UdpServer(server_address='localhost', server_port=6340)
target_address = 'localhost'
target_port = 6341

seq_len = 30
n_of_features = 2

input_buf = np.zeros(shape=(1, seq_len, n_of_features), dtype=np.float32)

i =0
while True:
    i += 1
    read_msg = udp_handle.receive_msg()
    input_sig = np.frombuffer(read_msg, dtype=np.float32)

    input_buf = np.roll(a=input_buf, shift=-1, axis=1)
    input_buf[0, -1] = input_sig

    est_flow = np.squeeze(model.run(output_names=None, input_feed={'input': input_buf})).item()
    output_msg = struct.pack('f', est_flow)
    udp_handle.send_msg(message=output_msg, client_address=target_address, port=target_port)