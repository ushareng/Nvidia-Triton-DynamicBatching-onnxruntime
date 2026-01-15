import onnx
from onnx import helper, TensorProto
import os

X = helper.make_tensor_value_info("INPUT", TensorProto.FLOAT, ["B", 4])
Y = helper.make_tensor_value_info("OUTPUT", TensorProto.FLOAT, ["B", 4])

two = helper.make_tensor("TWO", TensorProto.FLOAT, [1], [2.0])
node = helper.make_node("Mul", inputs=["INPUT", "TWO"], outputs=["OUTPUT"])

graph = helper.make_graph([node], "double_graph", [X], [Y], initializer=[two])
model = helper.make_model(graph, producer_name="demo_double")
model.ir_version = 11
model.opset_import[0].version = 13

out_dir = os.path.expanduser("~/model_repository/onnx_double/1")
os.makedirs(out_dir, exist_ok=True)
onnx.save(model, os.path.join(out_dir, "model.onnx"))

print("ONNX model saved")
