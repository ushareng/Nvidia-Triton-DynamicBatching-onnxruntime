# Triton Dynamic Batching ONNX Demo (CPU)

This repository demonstrates **dynamic batching** using **NVIDIA Triton Inference Server** on a **CPU-only Ubuntu VM**, without Docker or GPUs.

The demo uses a minimal **ONNX Runtime** model that doubles the input tensor values and shows how Triton batches multiple concurrent requests into fewer executions.

<img width="1907" height="256" alt="image" src="https://github.com/user-attachments/assets/d332f42a-28e8-4755-a0db-91516f450292" />

---

## 🚀 What This Demo Shows

- Running Triton Inference Server **natively on Ubuntu**
- Deploying an **ONNX model** using the `onnxruntime` backend
- Enabling and validating **dynamic batching**
- Sending **concurrent gRPC requests**
- Verifying batching behavior via **Prometheus metrics**

---

## 🧠 Model Overview

- **Model name:** `onnx_double`
- **Backend:** ONNX Runtime
- **Operation:** `OUTPUT = INPUT * 2`
- **Input shape:** `[-1, 4]` (dynamic batch dimension)
- **Device:** CPU

---

## 📁 Repository Structure

```text
.
├── model_repository/
│   └── onnx_double/
│       ├── 1/
│       │   └── model.onnx
│       └── config.pbtxt
├── client_dynbatch_onnx.py
├── generate_onnx_model.py
└── README.md
```

---

## ⚙️ Dynamic Batching Configuration

```pbtxt
dynamic_batching {
  preferred_batch_size: [ 2, 4, 8 ]
  max_queue_delay_microseconds: 2000
}
```

---

## ▶️ Running Triton Server

```bash
./bin/tritonserver   --model-repository=$HOME/model_repository   --http-port=8000   --grpc-port=8001   --metrics-port=8002
```

---

## 🧪 Running the Client

```bash
pip install -U tritonclient[grpc] numpy
python3 client_dynbatch_onnx.py
```

---

## 📊 Verify Dynamic Batching

```bash
curl localhost:8002/metrics | grep onnx_double
```

Look for:
- `nv_inference_request_success`
- `nv_inference_exec_count` < request count

---

## 📌 Requirements

- Ubuntu 22.04+
- Python 3.10+
- Triton Inference Server 2.64.0
- CPU-only (no GPU required)

---

## 📄 License

This project is licensed under the MIT License.

© 2026 Usha Rengaraju

See the [LICENSE](LICENSE) file for full details.
