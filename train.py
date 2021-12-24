"""
train Q-Bot with referee Q-Bot
add entity information
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
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot
from eval_utils.rank_questioner import rankQABots
from eval_utils.rank_questioner import rankQABots_with_guesser
from eval_utils.dialog_generate import run_dialog
from utils import utilities as utils
from utils.visualize import VisdomVisualize

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
ind2word = dataset.ind2word
to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
    lambda x:x>0,w.data.cpu().numpy()))][:l.item()])) #.encode('utf-8','ignore')

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
aBot = None
qBot = None

# Loading Q-Bot
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
    for key in loadedParams:
        params[key] = loadedParams[key]

    if params['trainMode'] == 'rl-full-QAf' and params['freezeQFeatNet']:
        qBot.freezeFeatNet()
    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))


# Looading Guesser
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    guesser, _, _ = utils.loadModel(params, 'guesser')
    guesser.eval()


# Loading A-Bot
if params['trainMode'] in ['sl-abot', 'rl-full-QAf']:
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot')
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())

# Loading A-Bot for eval
if aBot is None:
    assert params['startFrom']!=''
    eval_aBot, _, _ = utils.loadModel(params, 'abot')
else:
    eval_aBot = aBot

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
running_KLLoss = None

mse_criterion = nn.MSELoss(reduce=False)
KL_criterion = nn.KLDivLoss(reduction='none')

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - ((startIterID - 1) // numIterPerEpoch)%(9 - params["annealingEndRound"]))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

start_t = timer()


baseline = 0

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

    n_batch = caption.size(0)


    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    cos_similarity_loss = 0
    huber_loss = 0
    rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    KLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']

    question_list = [[] for j in range(n_batch)]
    answer_list = [[] for j in range(n_batch)]
    reward_list = [[] for j in range(n_batch)]


    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
    if qBot:
        qBot.train(), qBot.reset()
        if params['use_entity']:
            entity_count = Variable(batch['entity_count'], requires_grad=False)
            entity_length = Variable(batch['entity_length'], requires_grad=False)
            entity_id = Variable(batch['entity_id'], requires_grad=False)  # (batch_size, round, n_entity, 3)
            entity_prob = Variable(batch['entity_prob'], requires_grad=False)
            qBot.observe(-1, caption=caption, captionLens=captionLens, image=image, entity_count=entity_count, entity_length=entity_length, entity_id=entity_id, entity_prob=entity_prob)
        else:
            qBot.observe(-1, caption=caption, captionLens=captionLens)

        if params['trainMode'] == 'rl-full-QAf' and params['use_new_rwd']:
            guesser.eval(), guesser.reset()
            guesser.observe(-1, caption=caption, captionLens=captionLens, image=image)


    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
        initialGuess = qBot.predictImage()
        prevFeatDist = mse_criterion(initialGuess, image)
        featLoss += torch.mean(prevFeatDist)
        prevFeatDist = torch.mean(prevFeatDist,1)

        if params['trainMode'] == 'rl-full-QAf':
            if params['use_new_rwd']:
                prevGuess = guesser.predictImage()
            else:
                prevGuess = qBot.predictImage()

    cum_reward = torch.zeros(params['batchSize'])
    if params['useGPU']:
        cum_reward = cum_reward.cuda()
    past_dialog_hidden = None
    cur_dialog_hidden = None

    mean_reward_batch = 0
    # calculate the mean reward value for this batch. This will be used to update baseline.
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:
            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers
            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)
            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                in mean squared error between the current and
                previous round of Q-Bot's image prediction.
                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.
        '''
        factRNN = None
        dialogRNN = None
        dialogState = None

        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound and params["AbotMCTS"]:

            factRNN = qBot.encoder.factRNN
            dialogRNN = qBot.encoder.dialogRNN
            dialogState = qBot.encoder.dialogHiddens[-1]

        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = (params['trainMode'] == 'sl-abot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot feature regression network
        forwardFeatNet = (forwardQBot or params['trainMode'] == 'rl-full-QAf')

        # Answerer Forward Pass
        if forwardABot:
            # Observe Ground Truth (GT) question
            aBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            # Observe GT answer for teacher forcing
            aBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])
            ansLogProbs = aBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Answers
            aBotLoss += utils.maskedNll(ansLogProbs,
                                        gtAnswers[:, round].contiguous())

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])

            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll(quesLogProbs,
                                        gtQuestions[:, round].contiguous())

            # input is log probability
            if params['use_entity'] and params['use_kl']:
                kl = KL_criterion(qBot.encoder.prior_logits, entity_prob[:, round, :]).sum(1)
                KLLoss += kl.mean()
                # KLLoss += (kl * ent_ques_mask).sum() / (ent_ques_mask.sum()+1e-10)

            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

            if params['trainMode'] == 'rl-full-QAf':
                if params['use_new_rwd']:
                    guesser.observe(round, ques=gtQuestions[:, round], quesLens=gtQuesLens[:, round])
                    guesser.observe(round, ans=gtAnswers[:, round], ansLens=gtAnsLens[:, round])
                for i in range(n_batch):
                    question_list[i].append(to_str_gt(gtQuestions[i][round])[8:-6])
                    answer_list[i].append(to_str_gt(gtAnswers[i][round])[8:-6])
                    reward_list[i].append(float('inf'))


        # A-Bot and Q-Bot interacting in RL rounds
        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound:
            # Run one round of conversation
            if params['use_entity']:
                questions, quesLens = qBot.forwardDecode(inference='sample', mode='rl')  # only Q-Bot need to set mode
            else:
                questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            if params["AbotMCTS"]:
                answers, ansLens = aBot.forwardDecode(inference='sample',run_mcts=True)
            else:
                answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            if params['use_new_rwd']:
                guesser.observe(round, ques=questions, quesLens=quesLens)
                guesser.observe(round, ans=answers, ansLens=ansLens)

                # Q-Bot makes a guess at the end of each round
                predFeatures = guesser.predictImage()
            else:
                predFeatures = qBot.predictImage()

            # Computing reward based on Q-Bot's predicted image
            reward = mse_criterion(prevGuess, image) - mse_criterion(predFeatures, image)
            reward = reward.mean(1).detach()

            reward = reward * params["rewardCoeff"]
            cum_reward = cum_reward + reward.data


            #TODO this should be an input; ignoring discount factor for now
            if params['rlAbotReward']:

                mean_reward_batch += float(torch.mean(reward))
                reward_q = reward
                reward_a = reward
                aBotRLLoss = aBot.reinforce(reward_a - baseline)
                qBotRLLoss = qBot.reinforce(reward_q - baseline)

            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)

            for i in range(n_batch):
                question_list[i].append(to_str_pred(questions[i], quesLens[i])[8:])
                answer_list[i].append(to_str_pred(answers[i], ansLens[i])[8:])
                reward_list[i] += [reward[i].item()]


            if idx%500==0 and round==9:
                for bc in range(2):
                    cap = to_str_gt(caption[bc])[8:-6]
                    print('C: ', cap.encode('ascii', 'ignore').decode('ascii'))
                    for rnd in range(numRounds):
                        ques = question_list[bc][rnd]
                        ans = answer_list[bc][rnd]
                        rwd_cur = reward_list[bc][rnd]
                        print('Q: ', ques.encode('ascii', 'ignore').decode('ascii'), ' ', 'A: ', ans.encode('ascii', 'ignore').decode('ascii'), '       rwd: %.3g' %rwd_cur)
                    print('######################################')



        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

            if params['trainMode'] == 'rl-full-QAf':
                if params['use_new_rwd']:
                    prevGuess = guesser.predictImage()
                else:
                    prevGues = qBot.predictImage()



    # baseline = (0.95 * baseline) + (0.05 * mean_reward_batch /
    #                                 (params["numRounds"] - rlRound))

    # Loss coefficients
    rlCoeff = params['RLLossCoeff']
    rlLoss = rlLoss * rlCoeff
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds
    KLLoss = (params['CELossCoeff'] * KLLoss) / numRounds

    featLoss = featLoss / numRounds  # / (numRounds+1)
    rlLoss = rlLoss / numRounds

    cos_similarity_loss = (params['CosSimilarityLossCoeff'] * cos_similarity_loss) / numRounds
    huber_loss = -(params["HuberLossCoeff"] * huber_loss)/numRounds
    avg_reward = torch.mean(cum_reward)

    loss = qBotLoss + aBotLoss + rlLoss + featLoss + KLLoss
    loss.backward()

    if params["clipVal"]:
        _ = nn.utils.clip_grad_norm_(parameters, params["clipVal"])

    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.item()


    # record history loss
    if qBot:
        qCE_history += [qBotLoss.item()]
        qloss_history += [loss.item()]

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        curEpoch = int(float(iterId) / numIterPerEpoch)
        if curEpoch % params['annealingReduceEpoch'] == 0:
            if params['trainMode'] == 'rl-full-QAf':
                rlRound = max(params["annealingEndRound"], rlRound - 1)
                if rlRound == params["annealingEndRound"]:
                    rlRound = params['numRounds'] - 1
                print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 100 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printFormat += '[CELoss: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.item(), lRate, qBotLoss.item()
        ]
        if params['use_entity'] and params['use_kl']:
            printFormat += '[KLLoss: %.3g]'
            printInfo += [KLLoss.item()]
        if params['trainMode']=='rl-full-QAf':
            printFormat += '[ALoss: %.3g][rwd: %.3g]'
            printInfo += [aBotLoss.item(), avg_reward.item()]

        
        start_t = end_t
        print(printFormat % tuple(printInfo))

    # Evaluate every epoch
    if (iterId + 1) % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID:w
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1
        
        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()
        if guesser:
            guesser.eval()

        if qBot:
            print("qBot Validation:")
            with torch.no_grad():
                rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val', num_workers=params['numWorkers'], params=params)

            for metric, value in rankMetrics.items():
                print(metric, value)


            # print train loss
            print(f'qBot train CE: {sum(qCE_history) / len(qCE_history)}')
            print(f'qBot train Loss: {sum(qloss_history) / len(qloss_history)}')
            qloss_history = []
            qCE_history = []

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                print("val CE", logProbsMean)

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])


            huberLossMean = 0
            if params["useHuberLoss"]:
                huberLossMean = params['HuberLossCoeff'] * (
                    rankMetrics['huberLossMean'])


            cosSimilarityLossMean = 0
            if params["useCosSimilarityLoss"]:
                cosSimilarityLossMean = params['CosSimilarityLossCoeff'] * (
                    rankMetrics['cosSimilarityLossMean'])

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    if params["useHuberLoss"]:
                        valLoss += huberLossMean
                    if params["useCosSimilarityLoss"]:
                        valLoss += cosSimilarityLossMean
                    print("valLoss", valLoss)


  
        print('Performing validation...')
        if aBot and 'ques' in batch:
            print("aBot Validation:")

            # NOTE: A-Bot validation is slow, so adjust exampleLimit as needed
            with torch.no_grad():
                rankMetrics = rankABot(
                    aBot,
                    dataset,
                    'val',
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None,useNDCG=params["useNDCG"])

            for metric, value in rankMetrics.items():
                print(metric, value)

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                print("val CE", logProbsMean)

                if params['trainMode'] == 'sl-abot':
                    valLoss = logProbsMean

        if qBot and eval_aBot:
            split = 'val'
            splitName = 'full Val - {}'.format(params['evalTitle'])
            with torch.no_grad():
                rankMetrics, roundRanks = rankQABots_with_guesser(
                    qBot, guesser, eval_aBot, dataset, split, beamSize=params['beamSize'], params=params)
            for metric, value in rankMetrics.items():
                print(metric, value)


   # Save the model after every epoch
    if (iterId + 1) % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate
        curEpoch = int(curEpoch)
        if aBot:
            if curEpoch>0 and curEpoch+1 > params['save_total_limit']:
                remove_file = os.path.join(params['savePath'],
                                        'abot_ep_%d.vd' % (curEpoch-params['save_total_limit']))
                os.remove(remove_file)
                print(f'Delete ckpt {remove_file}')

            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            if curEpoch>0 and curEpoch+1 > params['save_total_limit']:
                remove_file = os.path.join(params['savePath'],
                                        'qbot_ep_%d.vd' % (curEpoch-params['save_total_limit']))
                os.remove(remove_file)
                print(f'Delete ckpt {remove_file}')
            

            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)

        print("Saving visdom env to disk: {}".format(params["visdomEnv"]))
