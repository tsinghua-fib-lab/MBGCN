# MBGCN
This is our implementation of paper:

*Bowen Jin, Chen Gao, Xiangnan He, Depeng Jin and Yong Li. 2020. [Multi-behavior Recommendation with Graph Convolutional Networks.](http://bowenjin.me/Multi-behaviour%20Recommendation%20with%20Graph%20Convolutional%20Networks.pdf)  In SIGIR'20.*

**Please cite our SIGIR'20 paper if you use our codes. Thanks!**

```
@inproceedings{jin2020multi,
  title={Multi-behavior Recommendation with Graph Convolution Networks},
  author={Jin, Bowen and Gao, Chen and He, Xiangnan and Jin, Depeng and Li, Yong},
  booktitle={43nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020},
}
```

Author: Bowen Jin (jbw17@mails.tsinghua.edu.cn)



## Enviroments

- python
- pytorch
- numpy
- visdom



## Sampling

Construct positive and negative item pair for BPR loss by running:

```
cd Tmall
mkdir sample_file
cd ..
python sample.py --path Tmall
```



## Running

#### Visdom

Open a visdom port by running

```
visdom -port 33337
```

Make port forwarding and then you can visit localhost:33337 with explorer.

#### Pretrain

Train MF first by running:

```
bash MF.sh
```

#### Train

Change 'pretrain_path' in MBGCN.sh to the path where the best MF model located.

Train MBGCN by running:

```
bash MBGCN.sh
```



## Note

We change sampling method from sampling online using DataLoader with 8 workers to sampling offline and save the pairs in .txt in advance. As a result, with code here, all BPR-based method including our MBGCN will get better performance compared with the performance in our paper.









