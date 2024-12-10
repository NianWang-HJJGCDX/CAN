This is the official code for our paper "Capsule Attention Network for Hyperspectral Image Classification", which is accepted by Remote Sensing 2024. 

If you think our code is helpful, please consider to star this repository. Thanks!

<<<<<<< HEAD
My interests focus on machine learning (deep learning) and its applications to image enhancement, data clustering (classification), object recognition (tracking) etc. Please feel free to  communicate with me. If you are interested at my works, you can follow my [Homepage](https://nianwang-hjjgcdx.github.io/).
=======
My interests focus on machine learning (deep learning) and its applications to image enhancement, data clustering (classification), object recognition (tracking) etc. Please feel free to  communicate with me. If you are interested at my works, you can get more papers and codes at my [Homepage](https://nianwang-hjjgcdx.github.io/).
>>>>>>> 3ce31862fdeaa584752e63a4bb13f436fc170764

Now we detail the usage of this code!

# Our Configuration
All the experiments were conducted by  PyTorch framework on a PC with one RTX 2080 GPU. 

# Our Environment
Python (3.8)
PyTorch (1.12.0) with CUDA (11.6)
numpy (1.24.3)

# Usage
Run "main.py", you will start training our CAN. After half of the total epochs, we test our model after each five epochs. After training, the best quantitative results will be printed in the panel. Corresponding visual map will be saved in "./results". Moreover, we record the training loss and overall accuracy for each epoch in "./loss_record".

Our CAN achieves the state-of-the-art performance with extremely low calculation, which makes you can derectly run it by large-scale data set 
Houston2018. Note: for this data set, you can change to test only once during the traning  by changing
"if epoch % args.test_freq == 1 and epoch >= args.epochs * 0.5:"
into
"if epoch % args.epochs == 1:".

We provide the Indian_pines data set. If you want to test on other data sets, please put the data into "./datasets" and change the path at "--dataset" in "main.py". Moreover, if you want to show the average results and standard deviation, you can set "--num_run" in "main.py" to test one data set by multiple runs.

For commercial use, please contact us for permission. For any problem, please also feel free to contract us. Email: nianwang04@outlook.com.

# Acknowledge
This work  partly relies on the following paper. We sincerely think their outstanding contributions.

[1] Sabour S, Frosst N, Hinton G E. Dynamic routing between capsules[J]. Advances in neural information processing systems, 2017, 30.

# Citation
If you use our code, please cite our paper by

@Article{CAN,
<br>AUTHOR = {Wang, Nian and Yang, Aitao and Cui, Zhigao and Ding, Yao and Xue, Yuanliang and Su, Yanzhao},
<br>TITLE = {Capsule Attention Network for Hyperspectral Image Classification},
<br>JOURNAL = {Remote Sensing},
<br>VOLUME = {16},
<br>YEAR = {2024},
<br>NUMBER = {21},
<br>DOI = {10.3390/rs16214001}
<br>}

