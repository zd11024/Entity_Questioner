import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime

def readCommandLine(argv=None):
    parser = argparse.ArgumentParser(description='Train and Test the Visual Dialog model')

    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-inputImg', default=None, help='HDF5 file with image features')
    parser.add_argument('-inputQues', default=None, help='HDF5 file with preprocessed questions')
    parser.add_argument('-inputJson', default=None, help='JSON file with info and vocab')
    parser.add_argument('-inputDenseJson', default=None, help='JSON file with dense annotations')

    parser.add_argument('-cocoDir', default='',
                            help='Directory for coco images, optional')
    parser.add_argument('-cocoInfo', default='',
                            help='JSON file with coco split information')

    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-verbose', type=int, default=1,
                            help='Level of verbosity (default 1 prints some info)',
                            choices=[1, 2])
    parser.add_argument('-savePath', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-saveName', default='',
                            help='Name of save directory within savePath')
    parser.add_argument('-startFrom', type=str, default='',
                            help='Copy weights from model at this path')
    parser.add_argument('-qstartFrom', type=str, default='',
                            help='Copy weights from qbot model at this path')
    parser.add_argument('-continue', action='store_true',
                            help='Continue training from last epoch')
    parser.add_argument('-enableVisdom', type=int, default=0,
                            help='Flag for enabling visdom logging')
    parser.add_argument('-visdomEnv', type=str, default='',
                            help='Name of visdom environment for plotting')
    parser.add_argument('-visdomServer', type=str, default='127.0.0.1',
                            help='Address of visdom server instance')
    parser.add_argument('-visdomServerPort', type=int, default=8893,
                            help='Port of visdom server instance')

    #-------------------------------------------------------------------------
    # Model params for both a-bot and q-bot
    parser.add_argument('-randomSeed', default=32, type=int,
                            help='Seed for random number generators')
    parser.add_argument('-imgEmbedSize', default=512, type=int,
                            help='Size of the multimodal embedding')
    parser.add_argument('-imgFeatureSize', default=4096, type=int,
                            help='Size of the image feature')
    parser.add_argument('-embedSize', default=300, type=int,
                            help='Size of input word embeddings')
    parser.add_argument('-rnnHiddenSize', default=512, type=int,
                            help='Size of the LSTM state')
    parser.add_argument('-numLayers', default=2, type=int,
                            help='Number of layers in LSTM')
    parser.add_argument('-imgNorm', default=1, type=int,
                            help='Normalize the image feature. 1=yes, 0=no')

    parser.add_argument('-AbotMCTS', default=0, type=int,
                        help='Running Rollouts for rewards calculation for Abot. 1=yes, 0=no')

    # A-Bot encoder + decoder
    parser.add_argument('-encoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use')
    parser.add_argument('-decoder', default='gen',
                            help='Name of the decoder to use (gen)',
                            choices=['gen'])
    # Q-bot encoder + decoder
    parser.add_argument('-qencoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use')

    parser.add_argument('-qdecoder', default='gen',
                            help='Name of the decoder to use (only gen supported now)')

    #-------------------------------------------------------------------------
    # Optimization / training params
    parser.add_argument('-trainMode', default='rl-full-QAf',
                            help='What should train.py do?',
                            choices=['sl-abot', 'sl-qbot', 'rl-full-QAf'])
    parser.add_argument('-numRounds', default=10, type=int,
                            help='Number of rounds of dialog (max 10)')
    parser.add_argument('-batchSize', default=20, type=int,
                            help='Batch size (number of threads) '
                                    '(Adjust base on GPU memory)')
    parser.add_argument('-learningRate', default=1e-3, type=float,
                            help='Learning rate')
    parser.add_argument('-weight_decay', default=0, type=float)  # L2 penalty
    parser.add_argument('-minLRate', default=5e-5, type=float,
                            help='Minimum learning rate')
    parser.add_argument('-dropout', default=0.5, type=float, help='Dropout')
    parser.add_argument('-numEpochs', default=85, type=int, help='Epochs')
    parser.add_argument('-lrDecayRate', default=0.99997592083, type=float,
                            help='Decay for learning rate')
    parser.add_argument('-CELossCoeff', default=1, type=float,
                            help='Coefficient for cross entropy loss')
    parser.add_argument('-RLLossCoeff', default=1, type=float,
                            help='Coefficient for cross entropy loss')
    parser.add_argument('-useCosSimilarityLoss', default=0, type=int,
                            help='whether to use similarity loss')
    parser.add_argument('-CosSimilarityLossCoeff', default=0.1, type=float,
                            help='Coefficient for similarity loss')

    parser.add_argument('-useHuberLoss', default=0, type=int,
                            help='whether to use Huber loss')
    parser.add_argument('-HuberLossCoeff', default=1, type=float,
                            help='Coefficient for Huber loss')

    parser.add_argument('-featLossCoeff', default=1000, type=float,
                            help='Coefficient for feature regression loss')
    parser.add_argument('-useCurriculum', default=1, type=int,
                            help='Use curriculum or for RL training (1) or not (0)')
    parser.add_argument('-freezeQFeatNet', default=0, type=int,
                            help='Freeze weights of Q-bot feature network')
    parser.add_argument('-rlAbotReward', default=1, type=int,
                            help='Choose whether RL reward goes to A-Bot')
    parser.add_argument('-clipVal',default=0, type=int,help='clip value')
    parser.add_argument('-rewardCoeff', default=10000, type=float,
                            help='Coefficient for feature regression loss')

    # annealing params"
    parser.add_argument('-annealingEndRound', default=3, type=int, help='Round at which annealing ends')
    parser.add_argument('-annealingReduceEpoch',default=1,type=int, help='Num epochs at which annealing happens')

    # Other training environmnet settings
    parser.add_argument('-useGPU', action='store_true', help='Use GPU or CPU')
    parser.add_argument('-numWorkers', default=1, type=int,
                            help='Number of worker threads in dataloader')
    parser.add_argument('-multiGPU', action='store_true', help='Use multiGPU or singleGPU')

    #-------------------------------------------------------------------------
    # Evaluation params
    parser.add_argument('-beamSize', default=5, type=int,
                            help='Beam width for beam-search sampling')
    parser.add_argument('-evalModeList', default=[], nargs='+',
                            help='What task should the evaluator perform?',
                            choices=['ABotRank', 'QBotRank', 'QABotsRank', 'dialog','human_study', 'GuesserRank'])
    parser.add_argument('-evalSplit', default='val',
                            choices=['train', 'val', 'test'])
    parser.add_argument('-evalTitle', default='eval',
                            help='If generating a plot, include this in the title')
    parser.add_argument('-startEpoch',default=1, type=int,help='Starting epoch for evaluation')
    parser.add_argument('-endEpoch',default=1, type=int,help='Last epoch for evaluation')
    parser.add_argument('-useNDCG', action='store_true',
                            help='Whether to use NDCG in evaluation')
    parser.add_argument('-discountFactor',default=0.5,type=float,help="discount factor for future rewards")
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    parser.add_argument('-save_total_limit', default=10, type=int, help='Limit the total amount of checkpoints')
    parser.add_argument('-print_every', default=20, type=int)
    parser.add_argument('-dataroot', default='', type=str)

    #  ReeQ
    parser.add_argument('-use_entity', default=1, type=int, help='whether to use entity')
    parser.add_argument('-n_entity', default=100, type=int, help='the max number of candidate entities')
    parser.add_argument('-max_entity_length', default=3, type=int)
    parser.add_argument('-frequency_limit', default=1, type=int)
    parser.add_argument('-use_kl', default=1, type=int, help='use kl loss to optimize the estimator')
    parser.add_argument('-use_new_rwd', default=1, type=int, help='use the reward obtained by the independ guesser')
    parser.add_argument('-d_entity_emb', default=512, type=int, help='the embbeding size of entity')

    # AuG
    parser.add_argument('-guesser', default='hre-ques-lateim-hist', type=str)
    parser.add_argument('-measure', default='l2', type=str)
    parser.add_argument('-margin', default=0.1, type=float)
    parser.add_argument('-negative_samples', default='batch,similar', type=str)
    parser.add_argument('-guesserFrom', default='', type=str)
    # candidate image
    parser.add_argument('-use_candidate_image', default=0, type=int)
    parser.add_argument('-n_candidate', default=100, type=int, help='size of candidates for each image')


    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))

    if parsed['saveName']:
        # Custom save file path
        parsed['savePath'] = os.path.join(parsed['savePath'],
                                          parsed['saveName'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
        parsed['savePath'] = os.path.join(parsed['savePath'], timeStamp)
        parsed['savePath'] += '_{:0>6d}'.format(random.randint(0, 10e6))

    file2path = {
        'inputImg': 'visdial/data_img.h5', 
        'inputQues': 'visdial/chat_processed_data.h5',
        'inputJson': 'visdial/chat_processed_params.json',
        'inputDenseJson': 'visdial/visdial_1.0_val_dense_annotations.json',
    }
    for x in file2path:
        if parsed[x] is None:
            parsed[x] = os.path.join(parsed['dataroot'], file2path[x])

    if parsed['use_entity']:
        parsed['entity_dict_file'] = os.path.join(parsed['dataroot'], 'object_vocab.txt')
        for split in ['train', 'val', 'test']:
            parsed['entity_%s_file' % split] = os.path.join(parsed['dataroot'], ('entity_%s_%d.pt' % (split, parsed['n_entity'])) )

    if parsed['use_candidate_image']:
        for split in ['train', 'val']:
            parsed['candidate_%s_file' % split] = os.path.join(parsed['dataroot'], ('candidate_%s_%d.json' % (split, parsed['n_candidate'])) )

    parsed['useHistory'] = True
    parsed['useIm'] = 'late'
    return parsed
