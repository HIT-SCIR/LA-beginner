# Sequence Labeling
## 项目介绍
本部分基于ATIS和SNIPS数据集，将槽位填充任务转化为BIO序列标注任务并给出LSTM和BERT实现。

## 环境配置
```shell
conda create -n test python=3.6
conda activate test
pip install -r requirements.txt
```
## 运行方式
```shell
# BiLSTM version ATIS datset
python train.py \
       --data_dir=./data/ATIS \
	   --save_dir=./save/ATIS \
	   -g \
	   --num_epoch=50 \
	   --batch_size=16 \
	   --learning_rate=1e-3 \
	   --dropout_rate=0.4 \
	   --word_embedding_dim=64 \
	   --encoder_hidden_dim=128  
```
```shell
# BERT verion ATIS datset
python train_for_bert.py \
       --data_dir=./data/ATIS \
	   --save_dir=./save/ATIS \
	   -g \
	   --num_epoch=50 \
	   --batch_size=16 \
	   --learning_rate=1e-6 \
	   --dropout_rate=0.4
```