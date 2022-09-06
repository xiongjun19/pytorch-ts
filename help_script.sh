1. 环境准备
git clone https://github.com/xiongjun19/pytorch-ts.git 




docker run --gpus all -p 9982:9982 --name cnt_ll2 --ipc=host -it -v /u/jxiong/workspace/dalle2-laion:/workspace/pts  pts 

CUDA_VISIBLE_DEVICES=3 python example_inference.py dream
