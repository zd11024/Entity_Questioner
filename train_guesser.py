"""
train guesser with margin loss
"""
import os
import gc
import random
import pprint
import sys
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_questioner import rankGuesser
from utils import utilities as utils

# VSE-C: https://github.com/ExplorerFreda/VSE-C
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def l2_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).sqrt().t()
    return score

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure='l2', max_violation=False, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'l2':
            self.sim = l2_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.reduction=='mean':
            return cost_s.mean() + cost_im.mean()
        else:
            return cost_s.sum() + cost_im.sum()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# hardest negative
class CapOnlyContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure='l2', max_violation=False, reduction='mean'):
        super(CapOnlyContrastiveLoss, self).__init__()
        self.margin = margin
        if measure=='l2':
            self.sim = l2_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.reduction = reduction

    def forward(self, im, s, ex_s):
        # compute image-sentence score matrix
        scores = self.sim(im, ex_s)
        scores_orig = self.sim(im, s)
        diagonal = scores_orig.diag().contiguous().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        if self.reduction=='mean':
            return cost_s.mean()
        else:
            return cost_s.sum()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Read the command line options
params = options.readCommandLine()
print(params)

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val']

dataset = VisDialDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds', 'startToken', 'endToken']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
os.makedirs(params['savePath'], exist_ok=True)

# Loading Modules
parameters = []
guesser = None
guesser, loadedParams, optim_state = utils.loadModel(params, 'guesser')
for key in loadedParams:
    params[key] = loadedParams[key]
parameters.extend(filter(lambda p: p.requires_grad, guesser.parameters()))

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=True,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

# Initializing visdom environment for plotting data
pprint.pprint(params)

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate, weight_decay=params['weight_decay'])
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None


mse_criterion = nn.MSELoss(reduce=False)
margin_criterion = ContrastiveLoss(margin=params['margin'], max_violation=True, measure=params['measure'])
external_margin_criterion = CapOnlyContrastiveLoss(margin=params['margin'], max_violation=True, measure=params['measure'])

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

start_t = timer()


# history
qCE_history = []
qloss_history = []

for epochId, idx, batch in batch_iter(dataloader):
    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()
    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
            else v for key, v in batch.items()}

    image = Variable(batch['img_feat'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)
    captionLens = Variable(batch['cap_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    if params['use_candidate_image']:
        cand_feat = Variable(batch['cand_feat'], requires_grad=False)
    n_batch = caption.size(0)


    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    featLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']

    # observe caption
    guesser.train(), guesser.reset()
    guesser.observe(-1, caption=caption, captionLens=captionLens)

    initialGuess = guesser.predictImage()
    gtFeatureEnc = image
    prevFeatDist = 0
    if 'batch' in params['negative_samples']:
        prevFeatDist += margin_criterion(initialGuess, gtFeatureEnc)
    if 'similar' in params['negative_samples']:
        dist = torch.tensor([1. / params['n_candidate']] * params['n_candidate']).view(1, params['n_candidate']).expand(cand_feat.size(0), cand_feat.size(1))  # (n_batch, n_cand)
        n_sample = 8
        ns_idx = torch.multinomial(dist, num_samples=n_sample).cuda()
        ns_sample = torch.gather(cand_feat, dim=1, index=ns_idx.view(n_batch, n_sample, 1).expand(n_batch, n_sample, 4096)).view(n_batch * n_sample, 4096) # (n_batch * n_sample, 4096)
        prevFeatDist += external_margin_criterion(initialGuess, gtFeatureEnc, ns_sample)
    featLoss += prevFeatDist

    for round in range(numRounds):

        # Tracking components which require a forward pass
        # Q-Bot dialog model
        forwardQBot = True
        # Q-Bot feature regression network
        forwardFeatNet = True


        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            guesser.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            guesser.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = guesser.predictImage()
            featDist = 0

            if 'batch' in params['negative_samples']:
                featDist = margin_criterion(predFeatures, gtFeatureEnc)        
            if 'similar' in params['negative_samples']:
                dist = torch.tensor([1. / params['n_candidate']] * params['n_candidate']).view(1, params['n_candidate']).expand(cand_feat.size(0), cand_feat.size(1))  # (n_batch, n_cand)
                n_sample = 8
                ns_idx = torch.multinomial(dist, num_samples=n_sample).cuda()
                ns_sample = torch.gather(cand_feat, dim=1, index=ns_idx.view(n_batch, n_sample, 1).expand(n_batch, n_sample, 4096)).view(n_batch * n_sample, 4096) # (n_batch * n_sample, 4096)
                featDist += external_margin_criterion(predFeatures, gtFeatureEnc, ns_sample)
        
            featLoss += featDist


    # Loss coefficients
    featLoss = featLoss * params['featLossCoeff'] / numRounds
    loss = featLoss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.item()

    # record history loss
    qloss_history += [loss.item()]

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']

    # Print every now and then
    if iterId % 100 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printFormat += '[featLoss: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.item(), lRate, featLoss.item()
        ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

    # Evaluate every epoch
    if (iterId) % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID:w
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1
        
        if guesser:
            guesser.eval()
            print("Guesser Validation:")
            with torch.no_grad():
                rankMetrics, roundMetrics = rankGuesser(guesser, dataset, 'val', num_workers=params['numWorkers'], params=params)

            for metric, value in rankMetrics.items():
                print(metric, value)


            # print train loss
            print(f'guesser train Loss: {sum(qloss_history) / len(qloss_history)}')
            qloss_history = []
            # qCE_history = []

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                print("val CE", logProbsMean)

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    print("valLoss", valLoss)


    # Save the model after every epoch
    if (iterId + 1) % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if guesser:
            if curEpoch>0 and curEpoch+1 > params['save_total_limit']:
                remove_file = os.path.join(params['savePath'], 
                                            'guesser_ep_%d.vd' % (curEpoch-params['save_total_limit']))
                os.remove(remove_file)
                print(f'Delete ckpt {remove_file}')
            
            saveFile = os.path.join(params['savePath'],
                                    'guesser_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(guesser, optimizer, saveFile, params)

        print("Saving visdom env to disk: {}".format(params["visdomEnv"]))
