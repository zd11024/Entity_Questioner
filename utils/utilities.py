import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from six import iteritems


# Initializing weights
def initializeWeights(root, itype='xavier'):
    assert itype == 'xavier', 'Only Xavier initialization supported'

    for module in root.modules():
        # Initialize weights
        name = type(module).__name__
        # If linear or embedding
        if name in ['Embedding', 'Linear']:
            fanIn = module.weight.data.size(0)
            fanOut = module.weight.data.size(1)

            factor = math.sqrt(2.0 / (fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            module.weight.data.copy_(weight)
        elif 'LSTM' in name:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.0)
                else:
                    fanIn = param.size(0)
                    fanOut = param.size(1)

                    factor = math.sqrt(2.0 / (fanIn + fanOut))
                    weight = torch.randn(fanIn, fanOut) * factor
                    param.data.copy_(weight)
        else:
            pass

        # Check for bias and reset
        if hasattr(module, 'bias') and type(module.bias) != bool:
            module.bias.data.fill_(0.0)


def saveModel(model, optimizer, saveFile, params):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'params': params,
    }, saveFile)

def loadModel(params, agent='abot', overwrite=False):
    try:
        params['vocabSize']=11323
        return _loadModel(params, agent)
    except:
        params['vocabSize']=11322
        return _loadModel(params, agent)

def _loadModel(params, agent='abot', overwrite=False):
    if overwrite is False:
        params = params.copy()
    loadedParams = {}
    # should be everything used in encoderParam, decoderParam below
    encoderOptions = [
        'encoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'numLayers',
        'useHistory', 'useIm', 'imgEmbedSize', 'imgFeatureSize', 'numRounds',
        'dropout'
    ]
    decoderOptions = [
        'decoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'numLayers',
        'dropout'
    ]
    modelOptions = encoderOptions + decoderOptions

    mdict = None
    gpuFlag = params['useGPU']
    multiGPU = params['multiGPU']
    continueFlag = params['continue']
    numEpochs = params['numEpochs']

    if agent=='abot':
        startArg = 'startFrom'
    elif agent=='qbot':
        startArg = 'qstartFrom'
    elif agent=='guesser':
        startArg = 'guesserFrom'

    if continueFlag:
        assert params[startArg], "Can't continue training without a \
                                    checkpoint"

    # load a model from disk if it is given
    if params[startArg]:
        print('Loading model (weights and config) from {}'.format(
            params[startArg]))

        if gpuFlag:
            mdict = torch.load(params[startArg])
        else:
            mdict = torch.load(params[startArg],
                map_location=lambda storage, location: storage)

        # Model options is a union of standard model options defined
        # above and parameters loaded from checkpoint
        modelOptions = list(set(modelOptions).union(set(mdict['params'])))
        for opt in modelOptions:
            if opt=='vocabSize' or opt=='frequency_limit':
                continue
            if opt not in params:
                # Loading options from a checkpoint which are
                # necessary for continuing training, but are
                # not present in original parameter list.
                if continueFlag:
                    print("Loaded option '%s' from checkpoint" % opt)
                    params[opt] = mdict['params'][opt]
                    loadedParams[opt] = mdict['params'][opt]

            elif params[opt] != mdict['params'][opt]:
                # When continuing training from a checkpoint, overwriting
                # parameters loaded from checkpoint is okay.
                if continueFlag:
                    print("Overwriting param '%s'" % str(opt))
                    params[opt] = mdict['params'][opt]

        params['continue'] = continueFlag
        params['numEpochs'] = numEpochs
        params['useGPU'] = gpuFlag

        if params['continue']:
            assert 'ckpt_lRate' in params, "Checkpoint does not have\
                info for restoring learning rate and optimizer."
            loadedParams['ckpt_lRate'] = mdict['params']['ckpt_lRate']

    # assert False, "STOP right there, criminal scum!"

    # Initialize model class
    encoderParam = {k: params[k] for k in encoderOptions}
    decoderParam = {k: params[k] for k in decoderOptions}
    encoderParam['startToken'] = params['startToken']
    encoderParam['endToken'] = params['endToken']
    decoderParam['startToken'] = params['startToken']
    decoderParam['endToken'] = params['endToken']

    if agent == 'abot':
        encoderParam['type'] = params['encoder']
        decoderParam['type'] = params['decoder']
        encoderParam['isAnswerer'] = True
        from visdial.models.answerer import Answerer
        model = Answerer(encoderParam, decoderParam,multiGPU=multiGPU)

    elif agent == 'qbot':
        encoderParam['isAnswerer'] = False
        encoderParam['useIm'] = False
        encoderParam['type'] = params['qencoder']
        decoderParam['type'] = params['qdecoder']
        decoderParam['d_entity_emb'] = params['d_entity_emb']
        encoderParam['frequency_limit'] = params['frequency_limit']

        if params['qencoder']=='entity_encoder':
            from visdial.models.entity_questioner import Questioner
        else:
            from visdial.models.questioner import Questioner  
        model = Questioner(
            encoderParam,
            decoderParam,
            imgFeatureSize=encoderParam['imgFeatureSize'],multiGPU=multiGPU, params=params)

    elif agent == 'guesser':
        encoderParam['isAnswerer'] = False
        encoderParam['useIm'] = False
        encoderParam['type'] = 'hre-ques-lateim-hist'
        decoderParam['type'] = 'gen'

        from visdial.models.questioner import Questioner
        model = Questioner(
            encoderParam,
            decoderParam,
            imgFeatureSize=encoderParam['imgFeatureSize'],multiGPU=multiGPU)


    if params['useGPU']:
        model.cuda()

    for p in model.encoder.parameters():
        p.register_hook(clampGrad)
    
    if hasattr(model, 'decoder'):
        for p in model.decoder.parameters():
            p.register_hook(clampGrad)
    # NOTE: model.parameters() should be used here, otherwise immediate
    # child modules in model will not have gradient clamping

    # copy parameters if specified
    if mdict:
        model.load_state_dict(mdict['model'])
        optim_state = mdict['optimizer']
    else:
        optim_state = None
    return model, loadedParams, optim_state


def clampGrad(grad, limit=5.0):
    '''
    Gradient clip by value
    '''
    grad.data.clamp_(min=-limit, max=limit)
    return grad


def getSortedOrder(lens):
    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().reshape(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    if isinstance(sortedLen, Variable):
        sortedLen = sortedLen.data
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder


def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               initialState=None,
               returnStates=False):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)

    if initialState is not None:
        hx = initialState
        sortedHx = [x.index_select(dim=1, index=fwdOrder) for x in hx]
        assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None

    rnnModel.flatten_parameters()
    _, (h_n, c_n) = rnnModel(packedSeqInput, hx)

    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        c_n = c_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output

def maskedNll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)

    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb

        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    # Shifting gtSeq 1 token left to remove <START>
    padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
    padColumn = Variable(padColumn)
    target = torch.cat([gtSeq, padColumn], dim=1)[:, 1:]

    # Generate a mask of non-padding (non-zero) tokens
    mask = target.data.gt(0)
    loss = 0
    if isinstance(gtSeq, Variable):
        mask = Variable(mask, volatile=gtSeq.volatile)
    assert isinstance(target, Variable)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    # Mean sentence probs:
    # gtLogProbs = gtLogProbs/(mask.float().sum(1).reshape(-1,1))
    if returnScores:
        return (gtLogProbs * (mask.float())).sum(1)
    maskedLL = torch.masked_select(gtLogProbs, mask)
    # nll_loss = -torch.sum(maskedLL) / torch.sum(mask)
    nll_loss = -torch.sum(maskedLL) / seq.size(0)
    return nll_loss


def concatPaddedSequences(seq1, seqLens1, seq2, seqLens2, padding='right'):
    '''
    Concates two input sequences of shape (batchSize, seqLength). The
    corresponding lengths tensor is of shape (batchSize). Padding sense
    of input sequences needs to be specified as 'right' or 'left'

    Args:
        seq1, seqLens1 : First sequence tokens and length
        seq2, seqLens2 : Second sequence tokens and length
        padding        : Padding sense of input sequences - either
                         'right' or 'left'
    '''

    concat_list = []
    cat_seq = torch.cat([seq1, seq2], dim=1)
    maxLen1 = seq1.size(1)
    maxLen2 = seq2.size(1)
    maxCatLen = cat_seq.size(1)
    batchSize = seq1.size(0)
    for b_idx in range(batchSize):
        len_1 = seqLens1[b_idx].item()
        len_2 = seqLens2[b_idx].item()

        cat_len_ = len_1 + len_2
        if cat_len_ == 0:
            raise RuntimeError("Both input sequences are empty")

        elif padding == 'left':
            pad_len_1 = maxLen1 - len_1
            pad_len_2 = maxLen2 - len_2
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][pad_len_2:]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][pad_len_1:]

            else:
                cat_ = torch.cat([seq1[b_idx][pad_len_1:],
                                  seq2[b_idx][pad_len_2:]], 0)
            cat_padded = F.pad(
                input=cat_,  # Left pad
                pad=((maxCatLen - cat_len_), 0),
                mode="constant",
                value=0)
        elif padding == 'right':
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][:len_1]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][:len_1]

            else:
                cat_ = torch.cat([seq1[b_idx][:len_1],
                                  seq2[b_idx][:len_2]], 0)
                # cat_ = cat_seq[b_idx].masked_select(cat_seq[b_idx].ne(0))
            cat_padded = F.pad(
                input=cat_,  # Right pad
                pad=(0, (maxCatLen - cat_len_)),
                mode="constant",
                value=0)
        else:
            raise (ValueError, "Expected padding to be either 'left' or \
                                'right', got '%s' instead." % padding)
        concat_list.append(cat_padded.unsqueeze(0))
    concat_output = torch.cat(concat_list, 0)
    return concat_output

def cosinePenalty(tensor1, tensor2):
    """
    Calculates cosine similarity between two tensors of size batch x dim
    
    Returns:
        tensor1,tensor2 -- torch.Tensor(shape: batch x dim)
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity_scores = cos(tensor1, tensor2)
    return torch.mean(similarity_scores)

def huberPenalty(tensor1, tensor2, threshold=0.1):
    """
    Calculates huber (Smooth-L1 penalty) loss between two tensors of size batch x dim
    https://en.wikipedia.org/wiki/Huber_loss
    Returns:
        tensor1,tensor2 -- torch.Tensor(shape: batch x dim)
        threshold -- l2 penalty if absolute difference is less than threshold, l1 penalty otherwise
    """
    assert tensor1.shape == tensor2.shape
    batch, _ = tensor1.shape
    norm_differences = torch.abs(tensor1 - tensor2)
    l2_mask = norm_differences <= threshold
    norm_differences_new = 0.5 * norm_differences * norm_differences * (l2_mask == 1).float()
    l1_mask = norm_differences > threshold
    norm_differences_new = norm_differences_new + (((l1_mask ==1).float()) * (threshold *
                                    (norm_differences  - (0.5 * threshold))))
    return torch.sum(norm_differences_new) / batch
