# mypy: ignore-errors
"""One-off: export HHP-Net's SavedModel to models/pretrained/hhp_net.onnx. Needs tensorflow and tf2onnx (not project
dependencies); run it in a separate venv. HHP-Net: github.com/cantarinigiorgio/HHP-Net (MIT)."""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import tf2onnx

m = tf.saved_model.load("HHP-Net/models/model_w300_biwi")
f = m.signatures["serving_default"]
spec = (tf.TensorSpec((None,15,1), tf.float32, name="input_1"),)
@tf.function(input_signature=spec)
def run(x):
    o = f(input_1=x)
    return o["yaw"], o["pitch"], o["roll"]
tf2onnx.convert.from_function(run, input_signature=spec, opset=13, output_path="/home/claude/httm/models/pretrained/hhp_net.onnx")
print("ok")
