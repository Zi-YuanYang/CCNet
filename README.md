# CO3Net: Coordinate-Aware Contrastive Competitive Neural Network for Palmprint Recognition

This repository is a PyTorch implementation of CO3Net (accepted by IEEE Transactions on Instrumentation and Measurement). This paper can be downloaded at [this link](https://ieeexplore.ieee.org/document/10124928).

#### Abstract
Palmprint recognition achieves high discrimination for identity verification. Compared with handcrafted local texture descriptors, convolutional neural networks (CNNs) can spontaneously learn optimal discriminative features without any prior knowledge. To further enhance the features' representation and discrimination, we propose a coordinate-aware contrastive competitive neural network (CO$_3$Net) for palmprint recognition. To extract the multi-scale textures, CO$_3$Net consists of three parallel learnable Gabor filters (LGF)-based texture extraction branches that learn the discriminative and robust ordering features. Due to the heterogeneity of palmprints, the effects of different textures on the final recognition performance are inconsistent, and dynamically focusing on the textures is beneficial to the performance improvement. Then, CO$_3$Net introduces the attention modules to explore the spatial information, and selects more robust and discriminative textures. Specifically, coordinate attention is embedded into CO$_3$Net to adaptively focus on the important textures from the positional information. Since it is difficult for the cross-entropy loss to build a compact intra-class and separate inter-class feature space, the contrastive loss is employed to jointly optimize the network. CO$_3$Net is validated on four public datasets, and the results demonstrate the remarkable recognition performance of the proposed CO$_3$Net compared to other state-of-the-art methods.


#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2022mtcc,
  title={CO3Net: Coordinate-Aware Contrastive Competitive Neural Network for Palmprint Recognition},
  author={Yang, Ziyuan and Xia, Wenjun and Qiao, Yifan and Lu, Zexin and Zhang, Bob and Leng, Lu and Zhang, Yi},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```

#### Requirements
Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Training
After you prepare the training and testing texts, then you can directly run our training code as follows:

```
python train.py --id_num xxxx --train_set_file xxxx --test_set_file xxxx --des_path xxxx --path_rst xxxx
```

* batch_size: the size of batch to be used for local training. (default: ```1024```)
* epoch_num: the number of total training epoches. (default: ```3000```)
* temp: the value of the tempture in our contrastive loss. (default: ```0.07```)
* weight1: the weight of cross-entropy loss. (default: ```0.8```)
* weight2: the weight of contrastive loss. (default: ```0.2```)
* id_num: the number of ids in the dataset.
* gpu_id: the id of training gpu.
* lr: the inital learning rate. (default: ```0.001```)
* redstep: the step size of learning scheduler. (default: ```500```)
* test_interval: the interval of testing.
* save_interval: the interval of saving.
* train_set_file: the path of training text file.
* test_set_file: the path of testing text file.
* des_path: the path of saving checkpoints.
* path_rst: the path of saving results.

#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is cziyuanyang@gmail.com.

#### Reference
We refer to the following repositories:
* https://github.com/JonnyLewis/compnet
* https://github.com/houqb/CoordAttention
