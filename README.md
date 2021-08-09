# LA组入门攻略
- [LA组入门攻略](#la组入门攻略)
  - [Part0: 写在前面](#part0-写在前面)
  - [Part1: NLP基础](#part1-nlp基础)
  - [Part2: LA经典论文推荐](#part2-la经典论文推荐)
  - [Part3: 模型框架](#part3-模型框架)
  - [Part4: 项目实践](#part4-项目实践)
  - [其他](#其他)

## Part0: 写在前面
欢迎加入赛尔语义分析LA组！

本攻略旨在帮助你快速学习NLP基础知识，并对LA相关的研究方向有一个大体的认识。

本攻略分为以下四个部分：
* Part1：NLP基础，在学习这一部分前，希望你已经掌握了微积分、线性代数、概率论相关知识。这一部分会按照NLP的发展历程，介绍词嵌入、神经网络、预训练模型等NLP通用方法，为后续阅读和复现相关论文提供坚实的基础。
* Part2：LA经典论文，这一部分是由实验室老师和学长推荐的NLP和LA经典论文。通过阅读原论文，让你对经典的模型和方法有更深入细致的理解，同时对将来可能的研究方向有一个大体的了解。
* Part3：模型构建，这一部分提供了一个带注释的代码框架，来学习基于深度学习的NLP模型框架，及其如何进行数据处理、训练和预测。在这一部分后，你应当对如何实现模型有明确的思路。
* Part4：项目实践，通过完成这一部分的任务，来确保你已经初步掌握了设计和实现模型的能力。
* 其他：一些课外的学习资源以供参考。

在学习过程中请善用各类搜索引擎，欢迎大家提出宝贵意见或建议。

祝大家学习顺利！

## Part1: NLP基础
* CS224n
  * [2021课程主页](http://web.stanford.edu/class/cs224n/)
  * [2019课程录像](https://www.bilibili.com/video/BV1Eb411H7Pq?from=search&seid=14373694631452542823)
  * 学有余力请阅读Suggested Readings中，各个算法的原论文
  * 完成五个Project
    * [参考答案](./Part1/参考答案)
  * [历年总结课件](./Part1/总结课件)

## Part2: [LA经典论文推荐](https://docs.qq.com/sheet/DVFl0eUlvaXpPdXZR)
* 详细介绍可以参考标题文档
* LA组
  * [From static to dynamic word representations: a survey](http://ir.hit.edu.cn/~car/papers/icmlc2020-wang.pdf)
    * From static to dynamic word representations: a survey
    * 词嵌入综述
  * [A Survey on Spoken Language Understanding: Recent Advances and New Frontiers](https://arxiv.org/abs/2103.03095)
    * Libo Qin, Tianbao Xie, Wanxiang Che, Ting Liu
    * SLU综述
  * [A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding](https://aclanthology.org/D19-1214/)
    * Libo Qin, Wanxiang Che, Yangming Li, Haoyang Wen, Ting Liu
    * 任务型对话
  * [Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation](http://ir.hit.edu.cn/~jxu/jun_files/papers/AAAI2020-Jun%20Xu-KnowHRL.pdf)
    * Jun Xu, Haifeng Wang, Zhengyu Niu, Hua Wu, Wanxiang Che
    * 知识型对话
  * [CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)
    * Libo Qin, Minheng Ni, Yue Zhang, Wanxiang Che
    * 数据增强
  * [Consistency Regularization for Cross-Lingual Fine-Tuning](https://arxiv.org/abs/2106.08226)
    * Bo Zheng, Li Dong, Shaohan Huang, Wenhui Wang, Zewen Chi, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei
    * 数据增强
  * [Sequence-to-sequence data augmentation for dialogue language understanding](https://arxiv.org/abs/1807.01554)
    * Yutai Hou, Yijia Liu, Wanxiang Che, and Ting Liu
    * 数据增强
  * [Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network](https://atmahou.github.io/attachments/atma's_acl2020_FewShot.pdf )
    * Yutai Hou, Wanxiang Che, Yongkui Lai, Zhihan Zhou, Yijia Liu, Han Liu and Ting Liu. 
    * 小样本
  * [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)
    * Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou
    * 多模态
  * [A Distributed Representation-Based Framework for Cross-Lingual Transfer Parsing](http://people.csail.mit.edu/jiang_guo/papers/jair2016-clnndep.pdf)
    * Jiang Guo, Wanxiang Che, David Yarowsky, Haifeng Wang, Ting Liu
    * 跨语言模型
* 非LA组
  * Learning Method
    * [Confident Learning: Estimating Uncertainty in Dataset Labels](http://www.researchgate.net/publication/337005918_Confident_Learning_Estimating_Uncertainty_in_Dataset_Labels)
      * CG Northcutt
      * 置信学习
    * [Learning Active Learning from Data](https://papers.nips.cc/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)
      * Ksenia Konyushkova
      * 主动学习
    * [A Simple Framework for Contrastive Learning of Visual Representations](https://static.aminer.cn/storage/pdf/arxiv/20/2002/2002.05709.pdf)
      * Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
      * 对比学习
    * [Neural Transfer Learning for Natural Language Processing](https://aran.library.nuigalway.ie/bitstream/handle/10379/15463/neural_transfer_learning_for_nlp.pdf?sequence=1&isAllowed=y)
      * Sebastian Ruder
      * 迁移学习
    * [Model-agnostic meta-learning for fast adaptation of deep networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
      * Chelsea Finn, Pieter Abbeel, Sergey Levine
      * 元学习
    * [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
      * Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra
      * 小样本元学习
  * Network Structure
    * [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km&noteId=rkl2Q1Qi6X&noteId=rkl2Q1Qi6X)
      * Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
      * 图神经网络
    * [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
      * Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
    * [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
      * Diederik P Kingma, Max Welling
      * VAE模型
    * [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
      * Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
      * GAN模型
    * [Pointer Networks](https://arxiv.org/abs/1506.03134)
      * Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
      * 指针网络
  * Knowledge Distillation
    * [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
      * Geoffrey Hinton, Oriol Vinyals, Jeff Dean
      * 蒸馏方法开山之作
    * [Sequence-Level Knowledge Distillation](https://arxiv.org/pdf/1606.07947.pdf?__hstc=36392319.43051b9659a07455a3db8391a8f20ea4.1480118400085.1480118400086.1480118400087.1&__hssc=36392319.1.1480118400088&__hsfp=528229161)
      * Yoon Kim, Alexander M. Rush
  * Cross-Linguistic
    * [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/)
      * Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov
    * [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)
      * Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou
  * Representation Learning(Pre-Trained Model)
    * [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
      * Yoshua Bengio,Réjean Ducharme,Pascal Vincent,Christian Jauvin
    * [Distributed Representations of Words and Phrases and their Compositionality](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
      * Tomas Mikolov,Ilya Sutskever,Kai Chen 
    * [Deep contextualized word representations](https://arxiv.org/abs/1409.0473)
      * Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
    * [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)
      * Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz & Samy Bengio
    * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
      * Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
      * BERT模型
    * [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)
      * Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning
      * ELECTRA模型
    * [Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066.pdf)
      * Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel
    * [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      * Alec Radford, Jeff Wu, R. Child, David Luan, Dario Amodei, Ilya Sutskever 
      * GPT2
    * [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
      * Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
      * XLNet模型
    * [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
      * Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
  * Prompt-based Laerning
    * [Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20/)
      * Timo Schick, Hinrich Schütze
    * [It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://aclanthology.org/2021.naacl-main.185.pdf)
      * Timo Schick, Hinrich Schütze
  * MultiModal Machine Learning(MMML)
    * [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)
      * Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee
    * [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165)
      * Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao
    * [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409)
      * Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, Haifeng Wang
    * [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)
      * Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer
  * Lexical Analysis
    * [Conditional random fields: Probabilistic models for segmenting and labeling sequence data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
      * John Lafferty, Andrew McCallum, Fernando Pereira
    * [Neural architectures for named entity recognition](https://aclanthology.org/N16-1030.pdf)
      * Neural Architectures for Named Entity Recognition
  * Syntactic Parsing
    * [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
      * Timothy Dozat, Christopher D. Manning
      * Biaffine模型
    * [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082/)
      * Danqi Chen,Christopher D. Manning
    * [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](https://aclanthology.org/P15-1033/)
      * Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, Noah A. Smith
  * Semantic Parsing
    * [Climbing towards NLU- On Meaning, Form, and Understanding in the Age of Data](https://aclanthology.org/2020.acl-main.463/)
      * Emily M. Bender, Alexander Koller
    * [Coarse-to-Fine Decoding for Neural Semantic Parsing](https://aclanthology.org/P18-1068.pdf)
      * Li Dong, Mirella Lapata
    * [A Syntactic Neural Model for General-Purpose Code Generation](https://aclanthology.org/P17-1041.pdf)
      * Pengcheng Yin, Graham Neubig
    * [Compositional Semantic Parsing on Semi-Structured Tables](https://aclanthology.org/P15-1142.pdf)
      * Panupong Pasupat, Percy Liang
  * Grammatical Error Correction(GEC)
    * [Encode, Tag, Realize: High-Precision Text Editing](https://arxiv.org/pdf/1909.01187.pdf)
      * Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, Aliaksei Severyn
    * [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
      * Xingyi Cheng, Weidi Xu, Kunlong Chen, Shaohua Jiang, Feng Wang, Taifeng Wang, Wei Chu, Yuan Qi
  * Dialogue
    * [Task-Oriented Dialogue as Dataflow Synthesis](https://arxiv.org/pdf/2009.11423.pdf)
      * Microsoft Semantic Machines
    * [Meena-Towards a Human-like Open-Domain Chatbot](https://arxiv.org/pdf/2001.09977.pdf)
      * Daniel Adiwardana, Minh-Thang Luong
    * [POMDP-based Statistical Spoken Dialogue Systems: a Review](https://www.microsoft.com/en-us/research/publication/pomdp-based-statistical-spoken-dialogue-systems-a-review/)
      * Steve Young Milica Gasic Blaise Thomson Jason Williams
      * 对话系统综述
    * [MultiWOZ -- A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://arxiv.org/abs/1810.00278)
      * Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, Milica Gašić
      * 对话领域影响力最大的数据集
    * [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536)
      * Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan
  * Question Answering(QA)
    * [Neural Reading Comprehension and Beyond](https://www.cs.princeton.edu/~danqic/papers/thesis.pdf)
      * Danqi Chen
    * [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
      * Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
  * Neural Machine Translation(NMT)
    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
      * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
      * Transformer模型
    * [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
      * Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    * [Neural machine translation of rare words with subword units.](https://www.aclweb.org/anthology/P16-1162.pdf)
      * Rico Sennrich, Barry Haddow, Alexandra Birch
      * 如何解决未登录词（OOV）
    * [Vocabulary Learning via Optimal Transport for Machine Translation](https://arxiv.org/abs/2012.15671)
      * Jingjing Xu, Hao Zhou, Chun Gan, Zaixiang Zheng, Lei Li
      * 如何找到词表合适的大小
    * [Levenshtein Transformer](https://arxiv.org/abs/1905.11006)
      * Jiatao Gu, Changhan Wang, Jake Zhao
  * Sentiment Analysis
    * [Document Modeling with Gated Recurrent Neural Network for Sentiment Classification](https://aclanthology.org/D15-1167.pdf)
      * Duyu Tang, Bing Qin, Ting Liu

## Part3: [模型框架](./Part3/BiLSTM-Seqlabeling)
* POS Tagging任务
* 基于BiLSTM
* 包含了数据处理、搭建、训练、测试的完整过程
* 对基于深度学习的NLP框架有一个大体的认识，为完成Part4做准备

## Part4: 项目实践
* 任务待定
  * 暂定NER和Graph-based Paser
  * 基于非预训练模型（#）
  * 基于预训练模型（#）
  * 计算资源？

## 其他
* 机器学习基础
  * 吴恩达
* 一些推荐的博客/github库
