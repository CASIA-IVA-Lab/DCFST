Learning Feature Embeddings for Discriminant Model based Tracking (Accepted by ECCV2020)

Abstract: After observing that the features used in most online discriminatively trained trackers are not optimal, in this paper, we propose a novel and effective architecture to learn optimal feature embeddings for online discriminative tracking. Our method, called DCFST, integrates the solver of a discriminant model that is differentiable and has a closed-form solution into convolutional neural networks. Then, the resulting network can be trained in an end-to-end way, obtaining optimal feature embeddings for the discriminant model-based tracker. As an instance, we apply the popular ridge regression model in this work to demonstrate the power of DCFST. Extensive experiments on six public benchmarks, OTB2015, NFS, GOT10k, TrackingNet, VOT2018, and VOT2019, show that our approach is efficient and generalizes well to class-agnostic target objects in online tracking, thus achieves state-of-the-art accuracy, while running beyond the real-time speed.

DCFST is a simple yet effective tracker. This project provides the training and test codes of DCFST. It produces the accuracy on six popular tracking benchmarks, OTB2015, NfS, TrackingNet, GOT10k, VOT2018, and VOT2019. Specifically, DCFST with resnet-18, i.e. DCFST-18, obtains the AUC score of 0.709 on OTB2015, 0.634 on NFS, 0.610 on GOT10k, 0.739 on TrackingNet, 0.416 on VOT2018, and 0.361 on VOT2019, DCFST with resnet-50, i.e. DCFST-50, obtains the AUC score of 0.641 on NFS, 0.638 on GOT10k, 0.752 on TrackingNet, 0.452 on VOT2018. Under the following hardwares, DCFST-18 and DCFST-50 run around 35FPS and 25FPS, respectively. I will gradually add parts of the code as soon as I have free time to deal with this project because I am very busy recently.

Thanks for the handsome work, ATOM, by Martin Danelljan. This project is highly based on pytracking (https://github.com/visionml/pytracking). Particularly, it is based on the early version (pytorch 0.4.1) of pytracking (https://github.com/visionml/pytracking/tree/pytorch041). Therefore, if there are any missing documents and operation descriptions due to my carelessness, please refer to the corresponding section in pytracking.

Please don't be confused by what I call the DCFST as SBDT in the code. In fact, I originally named the article of this work as "A Siample Baseline for Deep Tracking", thus I write code with the algorithm name SBDT. DCFST was the later changed article title and algorithm name, however, I do not change the name from SBDT to DCFST in the code overall, probably because I am lazy.

Download the weights of DCFST-18: https://drive.google.com/file/d/1MdTSAEOlsb-MK4VsWM9RuT7ed_FIvDQf/view?usp=sharing

Download the weights of DCFST-50: https://drive.google.com/file/d/12Ctc6oBz52CO6NyCWFOp3rrCxStILrzM/view?usp=sharing

Requirements: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz CPU; TITAN X(Pascal) or TITAN 1080Ti GPU; 8.0 CUDA; numpy  1.16.2; numpy-base  1.16.2; python  3.7.3; torchvision  0.2.1; pytorch  0.4.1; Pillow 5.4.1; pandas 1.0.3

*** NOTE *** I have tried my best to let everyone enjoy the above accuracy of DCFST accurately. The ATOM in pytracking is stochastic. In order to eliminate the randomness as much as possible, I make a small change to ATOM. For users, please ensure that the versions of softwares and environments you used are the same as above. There may be more version requirements and those are all I can think of so far.

Project in process ...
