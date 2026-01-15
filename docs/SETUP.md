# Setup Guide

## Start Triton Server
./bin/tritonserver \
  --model-repository=$HOME/model_repository \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002

## Verify model
curl localhost:8000/v2/models/onnx_double

## Run client
pip install tritonclient[grpc]
python client/client_dynbatch_onnx.py

## Verify dynamic batching
curl localhost:8002/metrics | grep onnx_double
