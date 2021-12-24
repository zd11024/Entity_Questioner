import os
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import random
from six import iteritems
from six.moves import range
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from typing import Dict, List, Union
def image_id_from_path(image_path):
    """Given a path to an image, return its id.
    Parameters
    ----------
    image_path : str
        Path to image, e.g.: coco_train2014/COCO_train2014/000000123456.jpg
    Returns
    -------
    int
        Corresponding image id (123456)
    """

    return int(image_path.split("/")[-1][-16:-4])

class VisDialDatasetHumanStudy(Dataset):
    def __init__(self, params, subsets):
        '''
            Initialize the dataset with splits given by 'subsets', where
            subsets is taken from ['train', 'val', 'test']
            Notation:
                'dtype' is a split taking values from ['train', 'val', 'test']
                'stype' is a sqeuence type from ['ques', 'ans']
        '''

        # By default, load Q-Bot, A-Bot and dialog options for A-Bot

        self.useQuestion = True
        self.useAnswer = True
        self.useOptions = True
        self.useHistory = True
        self.useIm = True
        self.useNDCG = params["useNDCG"]

        # Absorb parameters
        for key, value in iteritems(params):
            setattr(self, key, value)
        self.subsets = tuple(subsets)
        self.numRounds = params['numRounds']

        print('\nDataloader loading json file: ' + self.inputJson)
        with open(self.inputJson, 'r') as fileId:
            info = json.load(fileId)
            # Absorb values
            for key, value in iteritems(info):
                setattr(self, key, value)
        if 'val' in subsets and self.useNDCG:
            with open(self.inputDenseJson, 'r') as fileId:
                dense_annotation = json.load(fileId)
                self.dense_annotation = dense_annotation

        wordCount = len(self.word2ind)
        # Add <START> and <END> to vocabulary
        self.word2ind['<START>'] = wordCount + 1
        self.word2ind['<END>'] = wordCount + 2
        self.word2ind['<NULL>'] = wordCount + 3  # add <NULL>
        self.startToken = self.word2ind['<START>']
        self.endToken = self.word2ind['<END>']
        # Padding token is at index 0
        self.vocabSize = wordCount + 4
        print('Vocab size with <START>, <END>: %d' % self.vocabSize)

        # Construct the reverse map
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        # Read questions, answers and options
        print('Dataloader loading h5 file: ' + self.inputQues)
        quesFile = h5py.File(self.inputQues, 'r')

        if self.useIm:
            # Read images
            print('Dataloader loading h5 file: ' + self.inputImg)
            imgFile = h5py.File(self.inputImg, 'r')

        # Number of data points in each split (train/val/test)
        self.numDataPoints = {}
        self.data = {}

        # map from load to save labels
        ioMap = {
            'ques_%s': '%s_ques',
            'ques_length_%s': '%s_ques_len',
            'ans_%s': '%s_ans',
            'ans_length_%s': '%s_ans_len',
            'ans_index_%s': '%s_ans_ind',
            'img_pos_%s': '%s_img_pos',
            'opt_%s': '%s_opt',
            'opt_length_%s': '%s_opt_len',
            'opt_list_%s': '%s_opt_list'
        }

        # Processing id -> image_id 
        self.data['train_image_id'] = [image_id_from_path(image_fname) for image_fname in self.unique_img_train]
        self.data['val_image_id'] = [image_id_from_path(image_fname) for image_fname in self.unique_img_val]
        self.data['test_image_id'] = [image_id_from_path(image_fname) for image_fname in self.unique_img_test]
        # image_id -> id
        self.data['train_imgid_to_id'] = {img_id: i for i, img_id in enumerate(self.data['train_image_id'])}
        self.data['val_imgid_to_id'] = {img_id: i for i, img_id in enumerate(self.data['val_image_id'])}

        self.to_str_gt = lambda w: str(" ".join([self.ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())]))

        if self.use_entity:
            object_dict = {}
            stop_word = {'can', 'black', 'picture', 'trees', 'photo', 'day', 'sky', 'outside', 'animals', 'hair', 'buildings', 'water', 'grass', 'shirt', 'background', 'room', 'light', 'she', 'windows', 'night', 'yellow', 'floor', 'inside', 'lot', 'image', 'front'}

            with open(self.entity_dict_file) as f:
                for line in f.readlines():
                    line = line.strip()
                    if ',' in line:
                        splits=line.split(',')
                        for x in splits:
                            object_dict[x]=splits[0]
                    else:
                        if line in stop_word:
                            continue
                        object_dict[line]=line
                
                entity_all = [x for x in object_dict.values()]
                self.entity_all = entity_all

        # load train images
        imgFeats = np.array(imgFile['images_train'])
        # normalize, if needed
        print('Normalizing image features..')
        imgFeats = normalize(imgFeats, axis=1, norm='l2')

        # save img features
        self.data['train_img_fv'] = torch.FloatTensor(imgFeats)

        # Processing every split in subsets
        for dtype in subsets:  # dtype is in [train, val, test]
            print("\nProcessing split [%s]..." % dtype)
            if ('ques_%s' % dtype) not in quesFile:
                self.useQuestion = False
            if ('ans_%s' % dtype) not in quesFile:
                self.useAnswer = False
            if ('opt_%s' % dtype) not in quesFile:
                self.useOptions = False
            # read the question, answer, option related information
            for loadLabel, saveLabel in iteritems(ioMap):
                if loadLabel % dtype not in quesFile:
                    continue
                dataMat = np.array(quesFile[loadLabel % dtype], dtype='int64')
                self.data[saveLabel % dtype] = torch.from_numpy(dataMat)


            # Read image features, if needed
            if self.useIm:
                print('Reading image features...')
                imgFeats = np.array(imgFile['images_' + dtype])

                if not self.imgNorm:
                    continue
                # normalize, if needed
                print('Normalizing image features..')
                imgFeats = normalize(imgFeats, axis=1, norm='l2')

                # save img features
                self.data['%s_img_fv' % dtype] = torch.FloatTensor(imgFeats)
                # Visdial
                if hasattr(self, 'unique_img_train') and params['cocoDir']:
                    coco_dir = params['cocoDir']
                    with open(params['cocoInfo'], 'r') as f:
                        coco_info = json.load(f)
                    id_to_fname = {
                        im['id']: im['file_path']
                        for im in coco_info['images']
                    }
                    cocoids = getattr(self, 'unique_img_%s'%dtype)
                    if '.jpg' not in cocoids[0]:
                        img_fnames = [
                            os.path.join(coco_dir, id_to_fname[int(cocoid)])
                            for cocoid in cocoids
                        ]
                    else:
                        img_fnames = cocoids
                
                img_fnames = getattr(self, 'unique_img_%s'%dtype)
                self.data['%s_img_fnames' % dtype] = img_fnames
                
            if params['use_entity']:
                self.process_entity(dtype)

            # read the history, if needed
            if self.useHistory:
                captionMap = {
                    'cap_%s': '%s_cap',
                    'cap_length_%s': '%s_cap_len'
                }
                for loadLabel, saveLabel in iteritems(captionMap):
                    mat = np.array(quesFile[loadLabel % dtype], dtype='int32')
                    self.data[saveLabel % dtype] = torch.from_numpy(mat)

            # Number of data points
            self.numDataPoints[dtype] = self.data[dtype + '_cap'].size(0)


        # Prepare dataset for training
        for dtype in subsets:
            print("\nSequence processing for [%s]..." % dtype)
            self.prepareDataset(dtype)
        print("")

        # Default pytorch loader dtype is set to train
        # load image indices in the test set on which the human study is done. 
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]
        self.rand_idx = np.loadtxt('./data/human_study/human_study_indices.csv')
        self.NUM_SAMPLES = self.rand_idx.shape[0]
        
    def process_entity(self, dtype):
        print('Processing %s entity...'%dtype)
        if dtype=='train':
            entity_file=self.entity_train_file
        elif dtype=='val':
            entity_file=self.entity_val_file
        elif dtype=='test':
            entity_file=self.entity_test_file
        
        max_entity_length = self.max_entity_length  # max length

        entity_all=torch.load(entity_file)

        entity_token_eng = {}
        entity_token={}
        entity_count={}  # the number of entities retrieved by a caption
        entity_length={}  # the length of the entity

        for img_id in entity_all:
            entity_list=list(entity_all[img_id])
            entity_list=entity_list[:self.n_entity]
            if len(entity_list) < self.n_entity:
                entity_list += random.sample(self.entity_all, self.n_entity - len(entity_list))
            entity_list += ['<NULL>']
            entity_token_eng[img_id] = entity_list
            entity_count[img_id]=len(entity_list)

            entity_length_cur=torch.LongTensor(torch.Size([self.n_entity+1])).fill_(0)
            entity_token_cur=torch.LongTensor(torch.Size([self.n_entity+1, self.max_entity_length])).fill_(0)
            for i, e in enumerate(entity_list):
                tokens = e.split(' ')
                entity_length_cur[i]=len(tokens)
                for j, w in enumerate(tokens):
                   entity_token_cur[i,j]=self.word2ind.get(w, self.word2ind['UNK'])

            entity_token[img_id]=entity_token_cur
            entity_length[img_id]=entity_length_cur
        
        self.data['%s_entity_count'%dtype]=entity_count
        self.data['%s_entity_length'%dtype]=entity_length
        self.data['%s_entity_token'%dtype]=entity_token
        self.data['%s_entity_token_eng'%dtype]=entity_token_eng


    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    #----------------------------------------------------------------------------
    # Dataset preprocessing
    #----------------------------------------------------------------------------

    def prepareDataset(self, dtype):
        if self.useHistory:
            self.processCaption(dtype)

        # prefix/postfix with <START> and <END>
        if self.useOptions:
            self.processOptions(dtype)
            # options are 1-indexed, changed to 0-indexed
            self.data[dtype + '_opt'] -= 1

        # process answers and questions
        if self.useAnswer:
            self.processSequence(dtype, stype='ans')
            # 1 indexed to 0 indexed
            if dtype != 'test':
                self.data[dtype + '_ans_ind'] -= 1
        if self.useQuestion:
            self.processSequence(dtype, stype='ques')

    def processSequence(self, dtype, stype='ans'):
        '''
        Add <START> and <END> token to answers or questions.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
            'sentType' : Sequence type, either 'ques' or 'ans'
        '''
        assert stype in ['ques', 'ans']
        prefix = dtype + "_" + stype

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, numRounds, maxAnsLen = seq.size()
        newSize = torch.Size([numConvs, numRounds, maxAnsLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, :, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            for rId in range(numRounds):
                length = seqLen[thId, rId]
                if length == 0:
                    print('Warning: Skipping empty %s sequence at (%d, %d)'\
                          %(stype, thId, rId))
                    continue

                sequence[thId, rId, 1:length + 1] = seq[thId, rId, :length]
                sequence[thId, rId, length + 1] = endTokenId

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processCaption(self, dtype):
        '''
        Add <START> and <END> token to caption.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
        '''
        prefix = dtype + '_cap'

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, maxCapLen = seq.size()
        newSize = torch.Size([numConvs, maxCapLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            length = seqLen[thId]
            if length == 0:
                print('Warning: Skipping empty %s sequence at (%d)' % (stype, thId))
                continue

            sequence[thId, 1:length + 1] = seq[thId, :length]
            sequence[thId, length + 1] = endTokenId

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processOptions(self, dtype):
        ans = self.data[dtype + '_opt_list']
        ansLen = self.data[dtype + '_opt_len']

        ansListLen, maxAnsLen = ans.size()

        newSize = torch.Size([ansListLen, maxAnsLen + 2])
        options = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        options[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for ansId in range(ansListLen):
            length = ansLen[ansId]
            if length == 0:
                print('Warning: Skipping empty option answer list at (%d)'\
                        %ansId)
                continue

            options[ansId, 1:length + 1] = ans[ansId, :length]
            options[ansId, length + 1] = endTokenId

        self.data[dtype + '_opt_len'] = ansLen + 1
        self.data[dtype + '_opt_seq'] = options

    #----------------------------------------------------------------------------
    # Dataset helper functions for PyTorch's datalaoder
    #----------------------------------------------------------------------------

    def __len__(self):
        # Assert that loader_dtype is in subsets ['train', 'val', 'test']
        return self.NUM_SAMPLES

    def __getitem__(self, idx):

        print(type(self.rand_idx[idx]))
        idx = int(self.rand_idx[idx])
        item = self.getIndexItem(self._split, idx)
        return item

    def collate_fn(self, batch):
        out = {}

        mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}
        for key in mergedBatch:
            if key == 'img_fname' or key == 'index':
                out[key] = mergedBatch[key]
            elif key == 'cap_len' or key == 'cand_len' or key == 'cand_len_random' or key=='entity_count':
                # 'cap_lens' are single integers, need special treatment
                out[key] = torch.LongTensor(mergedBatch[key])
            else:
                out[key] = torch.stack(mergedBatch[key], 0)

        # Dynamic shaping of padded batch
        if 'ques' in out.keys():
            quesLen = out['ques_len'] + 1
            out['ques'] = out['ques'][:, :, :torch.max(quesLen)].contiguous()

        if 'ans' in out.keys():
            ansLen = out['ans_len'] + 1
            out['ans'] = out['ans'][:, :, :torch.max(ansLen)].contiguous()

        if 'cap' in out.keys():
            capLen = out['cap_len'] + 1
            out['cap'] = out['cap'][:, :torch.max(capLen)].contiguous()

        if 'opt' in out.keys():
            optLen = out['opt_len'] + 1
            out['opt'] = out['opt'][:, :, :, :torch.max(optLen) + 2].contiguous()

        return out

    #----------------------------------------------------------------------------
    # Dataset indexing
    #----------------------------------------------------------------------------

    def getIndexItem(self, dtype, idx):
        item = {'index': idx}

        # get question
        if self.useQuestion:
            ques = self.data[dtype + '_ques'][idx]
            quesLen = self.data[dtype + '_ques_len'][idx]
            item['ques'] = ques
            item['ques_len'] = quesLen

        # get answer
        if self.useAnswer:
            ans = self.data[dtype + '_ans'][idx]
            ansLen = self.data[dtype + '_ans_len'][idx]
            item['ans_len'] = ansLen
            item['ans'] = ans

        # get caption
        if self.useHistory:
            cap = self.data[dtype + '_cap'][idx]
            capLen = self.data[dtype + '_cap_len'][idx]
            item['cap'] = cap
            item['cap_len'] = capLen

        if self.useOptions and dtype != 'test':
            optInds = self.data[dtype + '_opt'][idx]
            ansId = self.data[dtype + '_ans_ind'][idx]

            optSize = list(optInds.size())
            newSize = torch.Size(optSize + [-1])

            indVector = optInds.view(-1)
            optLens = self.data[dtype + '_opt_len'].index_select(0, indVector)
            optLens = optLens.view(optSize)

            opts = self.data[dtype + '_opt_seq'].index_select(0, indVector)

            item['opt'] = opts.view(newSize)
            item['opt_len'] = optLens
            item['ans_id'] = ansId

        # if image needed
        if self.useIm:
            image_id = self.data['%s_image_id' % dtype][idx]

            item['img_feat'] = self.data[dtype + '_img_fv'][idx]

            # if image_id in self.data['%s_cand' % dtype]:
            #     cand_id = [img_id for img_id in self.data['%s_cand' % dtype][image_id] if image_id!=img_id]
            # else:
            #     cand_id = []

            # if len(cand_id) < self.n_candicate:
            #     cand_id += random.sample(self.data['train_image_id'], self.n_candicate - len(cand_id))

            # candicate_features = [self.data['train_img_fv'][self.data['train_imgid_to_id'][img_id]] for img_id in cand_id]

            # item['cand_id'] = torch.tensor(cand_id)
            # item['cand_len'] = len(candicate_features)
            # item['cand_feat'] = torch.stack(candicate_features)
            
            
            # # item['img_fname'] = self.data[dtype + '_img_fnames'][idx]
            # if dtype + '_img_labels' in self.data:
            #     item['img_label'] = self.data[dtype + '_img_labels'][idx]


        if self.use_entity:
            image_id=self.data['%s_image_id'%dtype][idx]
            
            item['entity_count']=self.data['%s_entity_count'%dtype][image_id]
            item['entity_length']=self.data['%s_entity_length'%dtype][image_id]
            item['entity_token']=self.data['%s_entity_token'%dtype][image_id]


            item['entity_prob'] = torch.zeros(self.numRounds, self.n_entity+1).float()
            item['entity_match_mask'] = torch.zeros(self.numRounds).float()
            for round in range(self.numRounds):
                ques = self.to_str_gt(item['ques'][round, :])[8:-6]
                ques_token = ques.split(' ')
                ques_token2 = [ques_token[i]+' '+ques_token[i+1] for i in range(len(ques_token)-1)]

                flg = False
                for i, kw in enumerate(self.data['%s_entity_token_eng'%dtype][image_id]):
                    if (kw in ques_token) or (kw in ques_token2):
                        item['entity_prob'][round, i] = 1
                        flg = True
                if flg:
                    item['entity_match_mask'][round] = 1
                else:
                    item['entity_match_mask'][round] = 0
                    item['entity_prob'][round, -1] = 1  # set <NULL> to 1                    


            item['entity_prob'] = F.normalize(item['entity_prob'], dim=-1, p=1)

        # dense annotations if val set
        if dtype == 'val' and self.useNDCG:

            round_id = self.dense_annotation[idx]['round_id']
            gt_relevance = self.dense_annotation[idx]['gt_relevance']
            image_id = self.dense_annotation[idx]['image_id']
            item["round_id"] = torch.LongTensor([round_id])
            item["gt_relevance"] = torch.FloatTensor(gt_relevance)
            item["image_id"] = torch.LongTensor([image_id])

        return item
        