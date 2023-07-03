# LA组入门攻略
## Part0: 写在前面
欢迎加入赛尔语言分析LA组！

本攻略旨在帮助你快速学习NLP基础知识，并对LA相关的研究方向有一个大体的认识。

本攻略分为以下四个部分：
* [Part1](#part1-nlp基础)：NLP基础，在学习这一部分前，希望你已经掌握了微积分、线性代数、概率论相关知识。这一部分会按照NLP的发展历程，介绍词嵌入、神经网络、预训练模型等NLP通用方法，为后续阅读和复现相关论文提供坚实的基础。
* [Part2](#part2-la经典论文推荐)：LA经典论文，这一部分是由实验室老师和学长推荐的NLP和LA经典论文。通过阅读原论文，让你对经典的模型和方法有更深入细致的理解，同时对将来可能的研究方向有一个大体的了解。
* [Part3](#part3-模型框架)：模型构建，这一部分提供了一个带注释的代码框架，来学习基于深度学习的NLP模型框架，及其如何进行数据处理、训练和预测。在这一部分后，你应当对如何实现模型有明确的思路。
* [Part4](#part4-项目实践)：项目实践，通过完成这一部分的任务，来确保你已经初步掌握了设计和实现模型的能力。
* [其他](#其他)：一些课外的学习资源以供参考。

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
* [《自然语言处理：基于预训练模型的方法》](https://item.jd.com/13344628.html)
  * 完成每一章的课后习题

## Part2: LA经典论文推荐
* 对于每一篇论文，对应的信息依次为：作者、会议/期刊、推荐理由
* LA组
  * [From static to dynamic word representations: a survey](http://ir.hit.edu.cn/~car/papers/icmlc2020-wang.pdf)
    * Yuxuan Wang, Yutai Hou, Wanxiang Che, Ting Liu
    * IJMLC 2020
    * 对静态词向量、动态词向量及其评价与应用做了很好地概述
  * [A Survey on Spoken Language Understanding: Recent Advances and New Frontiers](https://arxiv.org/abs/2103.03095)
    * Libo Qin, Tianbao Xie, Wanxiang Che, Ting Liu
    * IJCAI 2021 (Survey Track)
    * 任务型对话系统中SLU的首篇系统综述
  * [A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding](https://aclanthology.org/D19-1214/)
    * Libo Qin, Wanxiang Che, Yangming Li, Haoyang Wen, Ting Liu
    * EMNLP 2019
    * 任务型对话卓有成效的joint model
  * [Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation](http://ir.hit.edu.cn/~jxu/jun_files/papers/AAAI2020-Jun%20Xu-KnowHRL.pdf)
    * Jun Xu, Haifeng Wang, Zhengyu Niu, Hua Wu, Wanxiang Che
    * AAAI 2020
    * 提出主动式知识型对话建模框架和具体方法
  * [CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)
    * Libo Qin, Minheng Ni, Yue Zhang, Wanxiang Che
    * IJCAI 2020
    * 首次提出使用字典构造code-switching数据进行对齐多语言表示空间，受到谷歌，微软，facebook等大厂follow，并受谷歌大脑等企业邀请贡献核心代码。
  * [Consistency Regularization for Cross-Lingual Fine-Tuning](https://arxiv.org/abs/2106.08226)
    * Bo Zheng, Li Dong, Shaohan Huang, Wenhui Wang, Zewen Chi, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei
    * ACL 2021
    * 数据增强领域较为综合、阶段性的成果，从样本和模型两个层次做一致性建模，实验了多种常见数据增强模式，提升显著
  * [Sequence-to-sequence data augmentation for dialogue language understanding](https://arxiv.org/abs/1807.01554)
    * Yutai Hou, Yijia Liu, Wanxiang Che, and Ting Liu
    * COLING 2018
    * 数据增强
  * [Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network](https://atmahou.github.io/attachments/atma's_acl2020_FewShot.pdf )
    * Yutai Hou, Wanxiang Che, Yongkui Lai, Zhihan Zhou, Yijia Liu, Han Liu and Ting Liu. 
    * ACL 2020
    * 对话方向基于生成数据增强的工作
  * [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)
    * Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou
    * ACL 2021
    * 统一建模文本、图像和布局三种模态的信息，多个稳定理解数据集上SOTA结果
  * [A Distributed Representation-Based Framework for Cross-Lingual Transfer Parsing](http://people.csail.mit.edu/jiang_guo/papers/jair2016-clnndep.pdf)
    * Jiang Guo, Wanxiang Che, David Yarowsky, Haifeng Wang, Ting Liu
    * JAIR 2016
    * 包括了跨语言迁移的主要方法，对了解该领域很有帮助
* 非LA组
  * Learning Method
    * [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
      * Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra
      * NIPS 2016
      * 在AI 领域掀起 “小样本热潮“ 的开山工作之一，最经典基于metric learning的元学习工作之一
    * [Model-agnostic meta-learning for fast adaptation of deep networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
      * Chelsea Finn, Pieter Abbeel, Sergey Levine
      * ICML 2017
      * 最经典的基于optimization的元学习文章
    * [Learning Active Learning from Data](https://papers.nips.cc/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)
      * Ksenia Konyushkova
      * NIPS 2017
      * 主动学习方面代表作
    * [Neural Transfer Learning for Natural Language Processing](https://aran.library.nuigalway.ie/bitstream/handle/10379/15463/neural_transfer_learning_for_nlp.pdf?sequence=1&isAllowed=y)
      * Sebastian Ruder
      * Ph.D. thesis 2019
      * Sebastian Ruder的博士论文，第三章详细分析和比较了NLP中各种迁移学习方式
    * [A Simple Framework for Contrastive Learning of Visual Representations](https://static.aminer.cn/storage/pdf/arxiv/20/2002/2002.05709.pdf)
      * Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
      * ICML 2020
      * 不光适用于图像领域，在NLP中同样有用。对比学习学得的特征在下游任务上表现很好。
    * [Confident Learning: Estimating Uncertainty in Dataset Labels](http://www.researchgate.net/publication/337005918_Confident_Learning_Estimating_Uncertainty_in_Dataset_Labels)
      * CG Northcutt
      * JAIR 2021
      * 比较实用，在CV上得到了验证
  * Network Structure
    * [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
      * Diederik P Kingma, Max Welling
      * ICLR 2014
      * 提出 VAE，其中利用变分推断有效进行隐变量采样的技术成为经典，引领了后续一大批基于隐变量建模研究风格迁移、可控生成、表示学习等应用的工作
    * [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
      * Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
      * NIPS 2014
      * GAN 的第一篇文章，为对抗生成系列方法奠定基础
    * [Pointer Networks](https://arxiv.org/abs/1506.03134)
      * Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
      * NIPS 2015
      * Pointer Network 是一种经典的 seq2seq 模型，其中从输入复制到输出的机制成为了经典操作
    * [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
      * Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
      * ICLR 2018
      * Yoshua Bengio进军图神经网络
    * [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km&noteId=rkl2Q1Qi6X&noteId=rkl2Q1Qi6X)
      * Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
      * ICLR 2019
      * 引人思考图神经网络的强大之处
  * Knowledge Distillation
    * [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
      * Geoffrey Hinton, Oriol Vinyals, Jeff Dean
      * NIPS 2014
      * 蒸馏方法的开山之作
    * [Sequence-Level Knowledge Distillation](https://arxiv.org/pdf/1606.07947.pdf?__hstc=36392319.43051b9659a07455a3db8391a8f20ea4.1480118400085.1480118400086.1480118400087.1&__hssc=36392319.1.1480118400088&__hsfp=528229161)
      * Yoon Kim, Alexander M. Rush
      * EMNLP 2016
      * 将知识蒸馏成功用于NMT任务
  * Cross-Linguistic
    * [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)
      * Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou
      * ICLR 2018
      * 应用广泛的无监督跨语言词向量
    * [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/)
      * Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov
      * ACL 2020
      * 提出XLM-R，使跨语言预训练模型达到与单语相近的结果
    * [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)
      * Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer
      * TACL 2020
      * 多语言预训练的代表作之一。在多个对话任务，效果都很好
  * Representation Learning(Pre-Trained Model)
    * [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
      * Yoshua Bengio,Réjean Ducharme,Pascal Vincent,Christian Jauvin
      * JMLR 2003
      * 回顾bengio的经典论文：如何通过神经网络训练词向量
    * [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
      * Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
      * ICLR 2013
      * Word2Vec
    * [Distributed Representations of Words and Phrases and their Compositionality](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
      * Tomas Mikolov, Ilya Sutskever, Kai Chen
      * NIPS 2013
      * 这篇文章之后，词向量真正变成了一个现实的应用
    * [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf)
      * Jeffrey Pennington, Richard Socher, Christopher D. Manning
      * EMNLP 2014
      * 基于全局信息和上下文信息的词向量。
    * [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)
      * Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz & Samy Bengio
      * CoNLL 2016
      * 将VAE成功用于对自然语言的建模
    * [Deep contextualized word representations](https://arxiv.org/abs/1409.0473)
      * Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
      * NAACL 2018
      * NAACL2018 best paper
    * [Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066.pdf)
      * Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel
      * EMNLP 2019
      * 预训练语言模型中的知识
    * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
      * Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
      * NAACL 2019
      * 让pre-training + fine-tuning成为了NLP的新范式。
    * [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
      * Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
      * NIPS 2019
      * 对预训练方法的头脑风暴
    * [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      * Alec Radford, Jeff Wu, R. Child, David Luan, Dario Amodei, Ilya Sutskever 
      * Published 2019
      * GPT2
    * [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)
      * Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning
      * ICLR 2020
      * 巧妙的设计基于generator-discrimintor二分类的loss，有效提升预训练模型收敛速度及效果。
    * [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
      * Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
      * JMLR 2020
      * 提出了T5，用text-to-text的思想考虑所有NLP问题，论文详细介绍了设计决策依据。
  * Prompt-based Learning
    * [Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20/)
      * Timo Schick, Hinrich Schütze
      * EACL 2021
      * 提出cloze-style prompt-based fine-tuning方法Pet.
    * [It’s Not Just Size That Matters:Small Language Models Are Also Few-Shot Learners](https://aclanthology.org/2021.naacl-main.185.pdf)
      * Timo Schick, Hinrich Schütze
      * NAACL 2021
      * 第一次证明Prompt-based方法的威力
  * MultiModal Machine Learning(MMML)
    * [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)
      * Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee
      * NIPS 2019
      * Vision-and-Language 领域的经典双流模型
    * [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165)
      * Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao
      * ECCV 2020
      * 提出 Object-Semantics Aligned Pre-training，把物体用作视觉和语言语义层面上的定位点 ，以简化图像和文本之间的语义对齐的学习任务
    * [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409)
      * Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, Haifeng Wang
      * ACL 2021
      * 基于文本重写和文本/图像检索增强的跨模态对比学习来联合视觉和文本的语义空间，将多场景多模态数据作为输入，有效适应单模态和多模态的理解和生成任务
  * Lexical Analysis
    * [Conditional random fields: Probabilistic models for segmenting and labeling sequence data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
      * John Lafferty, Andrew McCallum, Fernando Pereira
      * ICML 2001
      * 把CRF 带入NLP 的工作。 理论和实际的结合非常值得学习。
    * [Neural architectures for named entity recognition](https://aclanthology.org/N16-1030.pdf)
      * Neural Architectures for Named Entity Recognition
      * NAACL 2016
      * NN序列标注之经典论文，简单易懂
  * Syntactic Parsing
    * [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082/)
      * Danqi Chen, Christopher D. Manning
      * EMNLP 2014
      * NN parser开山之作，简单易懂，入门必备
    * [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](https://aclanthology.org/P15-1033/)
      * Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, Noah A. Smith
      * ACL 2015
      * 提出Stack-LSTM，对基于转移的依存分析的发展有较大影响
    * [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
      * Timothy Dozat, Christopher D. Manning
      * ICLR 2017
      * 解决依存句法分析 Graph Based 的经典之作，Biaffine 模型不断在其他的 Parsing 任务上显现其能力
  * Semantic Parsing
    * [Compositional Semantic Parsing on Semi-Structured Tables](https://aclanthology.org/P15-1142.pdf)
      * Panupong Pasupat, Percy Liang
      * ACL 2015
      * 提出表格问答&语义解析经典数据集WikiTableQuestion
    * [A Syntactic Neural Model for General-Purpose Code Generation](https://aclanthology.org/P17-1041.pdf)
      * Pengcheng Yin, Graham Neubig
      * ACL 2017
      * 经典代码生成范式seq2tree：AST树的生成
    * [Coarse-to-Fine Decoding for Neural Semantic Parsing](https://aclanthology.org/P18-1068.pdf)
      * Li Dong, Mirella Lapata
      * ACL 2018
      * 经典代码生成范式seq2seq：先粗粒度后细粒度
    * [Climbing towards NLU- On Meaning, Form, and Understanding in the Age of Data](https://aclanthology.org/2020.acl-main.463/)
      * Emily M. Bender, Alexander Koller
      * ACL 2020
      * NLP人的“不忘初心”
  * Grammatical Error Correction(GEC)
    * [Encode, Tag, Realize: High-Precision Text Editing](https://arxiv.org/pdf/1909.01187.pdf)
      * Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, Aliaksei Severyn
      * EMNLP 2019
      * 创新性地将生成任务转换为text-editing任务，适合GEC等任务
    * [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
      * Xingyi Cheng, Weidi Xu, Kunlong Chen, Shaohua Jiang, Feng Wang, Taifeng Wang, Wei Chu, Yuan Qi
      * ACL 2020
      * 使用GCN模型将音近、形近信息引入BERT模型，CSC任务必读论文。
  * Dialogue
    * [POMDP-based Statistical Spoken Dialogue Systems: a Review](https://www.microsoft.com/en-us/research/publication/pomdp-based-statistical-spoken-dialogue-systems-a-review/)
      * Steve Young, Milica Gasic, Blaise Thomson, Jason Williams
      * IEEE Xplore 2013
      * 任务型对话的开山之作
    * [MultiWOZ -- A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://arxiv.org/abs/1810.00278)
      * Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, Milica Gašić
      * EMNLP 2018
      * 任务对话影响力最大的数据集
    * [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536)
      * Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan
      * ACL 2020
      * 在对话领域影响力很大的预训练模型
    * [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/pdf/2001.09977.pdf)
      * Daniel Adiwardana, Minh-Thang Luong, David R. So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, Quoc V. Le
      * Published 2020
      * 开放对话的里程碑
    * [Task-Oriented Dialogue as Dataflow Synthesis](https://arxiv.org/pdf/2009.11423.pdf)
      * Microsoft Semantic Machines组
      * TACL 2020
      * 有前途的任务对话语义理解
  * Question Answering(QA)
    * [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
      * Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
      * ICLR 2017
      * 抽取式阅读理解（如SQuAD）方向的经典论文，很多思想目前仍然沿用。
    * [Neural Reading Comprehension and Beyond](https://www.cs.princeton.edu/~danqic/papers/thesis.pdf)
      * Danqi Chen 
      * Ph.D. thesis 2018
      * 陈丹琦的毕业论文，建议阅读理解、开放域问答方向人士阅读。
  * Neural Machine Translation(NMT)
    * [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
      * Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
      * ICLR 2015 
      * Attention在NLP应用成功的开篇之作
    * [Neural machine translation of rare words with subword units.](https://www.aclweb.org/anthology/P16-1162.pdf)
      * Rico Sennrich, Barry Haddow, Alexandra Birch
      * ACL 2016
      * 处理OOV问题的经典方法，历久弥新
    * [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
      * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
      * NIPS 2017
      * 提出了Transformer这样一个效果优异的特征抽取器，被广泛应用于后续的预训练模型
    * [Levenshtein Transformer](https://arxiv.org/abs/1905.11006)
      * Jiatao Gu, Changhan Wang, Jake Zhao
      * NIPS 2019
      * 非自回归机器翻译经典模型
    * [Vocabulary Learning via Optimal Transport for Machine Translation](https://arxiv.org/abs/2012.15671)
      * Jingjing Xu, Hao Zhou, Chun Gan, Zaixiang Zheng, Lei Li
      * ACL 2021
      * ICLR转投ACL 2021 Best Paper，展示如何修改论文
  * Sentiment Analysis
    * [Document Modeling with Gated Recurrent Neural Networkfor Sentiment Classification](https://aclanthology.org/D15-1167.pdf)
      * Duyu Tang, Bing Qin, Ting Liu
      * EMNLP 2015
      * NN分类相对比较经典的论文，容易理解，适合入门

## Part3: [模型框架](./Part3/BiLSTM-Seqlabeling)
* Sequence-Labeling任务
* 基于BiLSTM
* 包含了数据处理、搭建、训练、测试的完整过程
* 对基于深度学习的NLP框架有一个大体的认识，为完成Part4做准备

## Part4: 项目实践
具体项目内容及要求见相关目录
* [Sequence Labeling](./Part4/Sequence_Labeling)
* [Graph-based Parser](./Part4/Graph-based%20Parser)

## 其他
* 机器学习基础
  * 吴恩达
* 一些推荐的博客/github库
