import numpy as np
import tritonclient.grpc as grpcclient
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "onnx_double"

def one_request(client, req_id):
    x = (np.ones((1, 4), dtype=np.float32) * req_id)
    inp = grpcclient.InferInput("INPUT", x.shape, "FP32")
    inp.set_data_from_numpy(x)
    out = grpcclient.InferRequestedOutput("OUTPUT")
    res = client.infer(MODEL, inputs=[inp], outputs=[out])
    y = res.as_numpy("OUTPUT")
    assert np.allclose(y, x * 2.0)

def main():
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    total = 200
    concurrency = 32

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, client, i) for i in range(total)]
        for f in as_completed(futures):
            f.result()

    print("All requests completed")

if __name__ == "__main__":
    main()
