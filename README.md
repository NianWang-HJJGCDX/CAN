This is the official code for our paper "Capsule Attention Network for Hyperspectral Image Classification", which is accepted by Remote Sensing 2024. 

If you think our code is helpful, please considering to star this repository. Thanks!

My interests focus on machine learning (deep learning) and its applications to image enhancement, data clustering (classification), object recognition (tracking) etc. Please feel free to  communicate with me. If you are interested at my works, you can follow my [ResearchGate] (https://www.researchgate.net/profile/Nian-Wang-19?ev=hdr_xprf&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImhvbWUiLCJwYWdlIjoiaG9tZSIsInBvc2l0aW9uIjoiZ2xvYmFsSGVhZGVyIn19)

Now we detail the usage of this code!

# Our Configuration
All the experiments were conducted by a PyTorch framework on a PC with one RTX 2080 GPU. 

# Our Environment
Python (3.8)
PyTorch (1.12.0) with CUDA (11.6)
numpy (1.24.3)

# Usage
Run "main.py", you will start training our CAN. After half of the total epochs, we test our model after each ten epochs. After training, the best quantitative results will be printed in the panel. Corresponding visual map will be saved in "./results". Moreover, we record the training loss and overall accuracy for each epoch in "./loss_record".

Our CAN achieves the state-of-the-art performance with extremely low calculation, which makes you can derectly run it by large-scale data set 
Houston2018. Note: for this data set, you can testing only once for acceleration by change 
"if epoch % args.test_freq == 1 and epoch >= args.epochs * 0.5:"
into
"if epoch % args.epochs == 1:".

We provide the Indian_pines data set. If you want to test on other data sets, please put the data into "./datasets" and change the path at "--dataset" in "main.py". Moreover, if you want to show the average results and standard deviation, you can set "--num_run" in "main.py" to test one data set by multiple runs.

For commercial use, please contact us for permission. For any problem, please also feel free to contract us. Email: nianwang04@outlook.com.

If you use our code, please cite our paper by

@Article{CAN,
AUTHOR = {Wang, Nian and Yang, Aitao and Cui, Zhigao and Ding, Yao and Xue, Yuanliang and Su, Yanzhao},
TITLE = {Capsule Attention Network for Hyperspectral Image Classification},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {21},
DOI = {10.3390/rs16214001}
}

