# Comprehensive Competition Mechanism in Palmprint Recognition
This repository is a PyTorch implementation of CCNet (accepted by IEEE Transactions on Information Forensics and Security). This paper can be accessed soon.

#### Abstract
Palmprint has gained popularity as a biometric modality and has recently attracted significant research inter est. The competition-based method is the prevailing approach for hand-crafted palmprint recognition, thanks to its powerful discriminative ability to identify distinctive features. However, the competition mechanism possesses vast untapped advantages that have yet to be fully explored. In this paper, we refor mulate the traditional competition mechanism and propose a Comprehensive Competition Network (CCNet). The traditional competition mechanism focuses solely on selecting the winner of different channels without considering the spatial information of the features. Our approach considers the spatial competition relationships between features while utilizing channel competition features to extract a more comprehensive set of competitive features. Moreover, existing methods for palmprint recognition typically focus on first-order texture features without utilizing the higher-order texture feature information. Our approach integrates the competition process with multi-order texture fea tures to overcome this limitation. CCNet incorporates spatial and channel competition mechanisms into multi-order texture features to enhance recognition accuracy, enabling it to capture and utilize palmprint information in an end-to-end manner efficiently. Extensive experimental results have shown that CCNet can achieve remarkable performance on four public datasets, showing that CCNet is a promising approach for palmprint recognition that can achieve state-of-the-art performance.

#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2023comprehensive,
  title={Comprehensive Competition Mechanism in Palmprint Recognition},
  author={Yang, Ziyuan and Huangfu, Huijie and Leng, Lu and Zhang, Bob and Teoh, Andrew Beng Jin and Zhang, Yi},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2023},
  publisher={IEEE}
}
```
