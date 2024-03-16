# CSR-STNet
Implementation of paper Colo-segment Recognition in Colonoscopy Videos via SpatioTemporal Network

## Requirements
- torch==2.0.1
- torchvision==0.15.2

## Instructions
Alter configs file to change model and training configurations. <br />
Use command `CUDA_VISIBLE_DEVICES=0 python main.py --config_path ./configs/default.yml` to train CSR-STNet.
