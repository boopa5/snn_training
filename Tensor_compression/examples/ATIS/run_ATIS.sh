
file=ATIS_tensor_2layers
use_cuda_graph=1
max_length=32


CUDA_VISIBLE_DEVICES=0 
python -u train_ATIS.py -save_model models/${file} -batch_size 16 -use_cuda_graph ${use_cuda_graph} -max_length ${max_length}|tee logs/${file}.txt
z# python -u ./examples/ATIS/train_ATIS.py -save_model models/ATIS_tensor_2layers -batch_size 16 -use_cuda_graph 1 -max_length 32|tee logs/ATIS_tensor_2layers.txt
# python -u train_ATIS.py -save_model examples/ATIS/models/ATIS_tensor_2layers -batch_size 16 -use_cuda_graph 0 -max_length 32|tee examples/ATIS/logs/ATIS_tensor_2layers.txt

# PATH=${PATH}:/usr/local/cuda-11.8/bin
# CUDA_HOEM=/usr/local/cuda-11.8