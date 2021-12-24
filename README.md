# Entity Questioner
The Code is pytorch implementation of `Enhancing Visual Dialog Questioner with Entity-based Strategy Learning and Augmented Guesser`.



## Env

1. Install python3.6 and pytorch=1.4.0.
2. Install requirements `pip install -r requirements.txt`.



## Data

1. Download preprocessed dataset and extracted features:

```shell
sh scripts/download_preprocessed.sh
```

2. Download the retrieved entity information (entity\_{split}\_100.pt) and candidate image information ( candidate\_{split}\_100.json) from https://drive.google.com/drive/folders/1s93pcmxHkPYMx9XXycFuZklC_oGeCqEJ?usp=sharing and palce them at the directory `./data`.  



## Training

We first train a Guesser agumented with negative samples and download the A-Bot provided in [Visdial-Diversity](https://github.com/vmurahari3/visdial-diversity). Since it is difficulty to independently investigate the QGen's performance, we indirectly evaluate QGen by observing the image-guessing performance (Guesser and A-Bot are fixed). Then, we train the QGen and A-Bot in RL.

### Guesser

```shell
sh scripts/train_guesser.sh <work_path> <save_name>
```

* <work_path>: the project path (default to .)
* <save_name>: the saved path

### QGen

```shell
sh scripts/train_qgen.sh <work_path> <save_name> <guesser_path> <abot_path>
```

* <guesser_path>: the pre-trained guesser
* <abot_path>: the pre-trained abot

### RL

```shell
sh scripts/train_rl.sh <work_path> <save_name> <guesser_path> <abot_path> <qbot_path>
```



## Evaluation

### Image-Guessing Performance

We evaluate the retrieval result under the image-guessing setting.

```shell
sh scripts/eval_sys <work_path> <guesser_path> <abot_path> <qbot_path>
```

* <work_path>: the project path (default to .)
* <guesser_path>: the path of Q-Bot
* <abot_path>: the path of A-Bot
* <qbot_path>: the path of Q-Bot

### A-Bot Performance

```shell
sh scripts/eval_abot.sh <work_path> <abot_path>
```

### Guesser Performance

We report the accuracy of guesser given the 10-round human dialogue.

```shell
sh scripts/eval_guesser.sh <work_path> <guesser_path>
```



## Citation

If you find this data is useful or use the data in your work, please cite our paper as well as Murahari et. al.,Improving Generative Visual Dialog by Answering Diverse Questions.

```
@misc{zheng2021enhancing,
      title={Enhancing Visual Dialog Questioner with Entity-based Strategy Learning and Augmented Guesser}, 
      author={Duo Zheng and Zipeng Xu and Fandong Meng and Xiaojie Wang and Jiaan Wang and Jie Zhou},
      year={2021},
      eprint={2109.02297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{murahari2019visdialdiversity,
  title={Improving Generative Visual Dialog by Answering Diverse Questions},
  author={Vishvak Murahari, Prithvijit Chattopadhyay, Dhruv Batra, Devi Parikh, Abhishek Das},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

Please contact Duo Zheng (zd[at].bupt.edu.cn) for questions and suggestions.

