# Comprehensive Competition Mechanism in Palmprint Recognition

This repository is a PyTorch implementation of CCNet (accepted by IEEE Transactions on Information Forensics and Security). This paper can be downloaded at [this link](https://ieeexplore.ieee.org/document/10223233).

#### Abstract
Palmprint has gained popularity as a biometric modality and has recently attracted significant research interest. The competition-based method is the prevailing approach for hand-crafted palmprint recognition, thanks to its powerful discriminative ability to identify distinctive features. However, the competition mechanism possesses vast untapped advantages that have yet to be fully explored. In this paper, we reformulate the traditional competition mechanism and propose a Comprehensive Competition Network (CCNet). The traditional competition mechanism focuses solely on selecting the winner of different channels without considering the spatial information of the features. Our approach considers the spatial competition relationships between features while utilizing channel competition features to extract a more comprehensive set of competitive features. Moreover, existing methods for palmprint recognition typically focus on first-order texture features without utilizing the higher-order texture feature information. Our approach integrates the competition process with multi-order texture features to overcome this limitation. CCNet incorporates spatial and channel competition mechanisms into multi-order texture features to enhance recognition accuracy, enabling it to capture and utilize palmprint information in an end-to-end manner efficiently. Extensive experimental results have shown that CCNet can achieve remarkable performance on four public datasets, showing that CCNet is a promising approach for palmprint recognition that can achieve state-of-the-art performance.


#### Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{yang2023ccnet,
  author={Yang, Ziyuan and Huangfu, Huijie and Leng, Lu and Zhang, Bob and Teoh, Andrew Beng Jin and Zhang, Yi},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Comprehensive Competition Mechanism in Palmprint Recognition}, 
  year={2023},
  volume={18},
  number={},
  pages={5160-5170},
  doi={10.1109/TIFS.2023.3306104}}
```

#### Requirements

If you have already tried our previous work [CO3Net](https://github.com/Zi-YuanYang/CO3Net), you can skip this step.

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Pretrained Model
To help readers use our model, we release our pretrained models for [Tongji](https://drive.google.com/file/d/1Kj6Q1eCpkbCbfPVSqTZHGV9Y_H1uToEM/view?usp=drive_link) and [IITD](https://drive.google.com/file/d/17EBrjGVrzcyjETobQCYYhfR87nD2Z-l6/view?usp=drive_link). Because this method is designed under a close-set setting, we encourage the readers to train our method in your dataset. 

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
* com_weight: the weight of the traditional competition mechanism. (default: ```0.8```)
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
* https://github.com/Zi-YuanYang/CO3Net
