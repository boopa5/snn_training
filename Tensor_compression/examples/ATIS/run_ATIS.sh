
file=ATIS_tensor_2layers
use_cuda_graph=1
max_length=32


CUDA_VISIBLE_DEVICES=0 python -u train_ATIS.py -save_model models/${file} -batch_size 16 -use_cuda_graph ${use_cuda_graph} -max_length ${max_length}|tee logs/${file}.txt


# PATH=${PATH}:/usr/local/cuda-11.8/bin
# CUDA_HOEM=/usr/local/cuda-11.8