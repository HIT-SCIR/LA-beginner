# Graph-based Parser
## 项目介绍
本部分基于PTBv3.0数据集，该数据为英文树库，[数据集介绍](https://cl.lingfil.uu.se/~nivre/research/Penn2Malt.html)，已预先标注POS，数据见`./data`文件夹。

## 任务介绍
使用深度学习方法，进行基于图的依存分析。
1. 基础任务：使用Biaffine模型进行基于图的依存分析，参考论文[《DEEP BIAFFINE ATTENTION FOR NEURAL DEPENDENCY PARSING》](https://arxiv.org/pdf/1611.01734.pdf)。
2. 进阶任务：尝试将Biaffine中的BiLSTM替换为BERT等预训练模型。
3. 高阶任务：基于图的依存分析中，时间开销最大的部分是根据评分进行解码。请任选一个深度学习模型，分别尝试[Eisner算法](https://aclanthology.org/C96-1058.pdf)和贪心算法（最小树形图），并尽可能优化时间开销。

## 备注
1. 基础任务和进阶任务为必选项，高阶任务为可选项。
2. 对于不熟悉预训练模型的同学，预训练模型部分建议使用HuggingFace框架。
3. 为方便对任务的理解，本攻略提供了一份基于BiLSTM的样例模型作为参考，具体使用方法见后文。

## 关于样例模型
### 环境准备
```shell
conda create -n parser python=3.9
conda activate parser
pip install -r requirements.txt
```

### 训练
```shell
python ./src/main.py \
       --max_line 1000000 \
       --batch_size 64 \
       --embedding_dim 128 \
       --hidden_size 256 \
       --layer_num 2 \
       --learning_rate 1e-5 \
       --epoch 300 \
       --core_num 3 \
       --decoder_type chuliu \
       --show_tqdm
```
模型保存在`./model`文件夹中，包括词表和训练模型。

### 测试
```shell
python ./src/main.py \
       --max_line 1000000 \
       --batch_size 32 \
       --mode test \
       --core_num 3 \
       --decoder_type chuliu \
       --show_tqdm
```
将会输出测试集上的评价指标。