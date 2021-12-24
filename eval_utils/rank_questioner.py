import sys
import json
import h5py
import gc
import numpy as np
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import options
import visdial.metrics as metrics
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from six.moves import range

from nltk.translate.bleu_score import sentence_bleu
import nltk
import collections
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from collections import Counter
from scipy.stats import entropy

def rankGuesser(qBot, dataset, split, exampleLimit=None, verbose=True, num_workers=1, beamSize=5, params=None):
    '''
        Evaluates Q-Bot performance on image retrieval when it is shown
        ground truth captions, questions and answers. Q-Bot does not
        generate dialog in this setting - it only encodes ground truth
        captions and dialog in order to perform image retrieval by
        predicting FC-7 image features after each round of dialog.

        Arguments:
            qBot    : Q-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn)

    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.item()])) #.encode('utf-8','ignore')

    # enumerate all gt features and all predicted features
    gtImgFeatures = []
    # caption + dialog rounds
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    logProbsAll = [[] for _ in range(numRounds)]
    featLossAll = [[] for _ in range(numRounds + 1)]
    start_t = timer()

    cos_similarity_loss = 0
    huber_loss = 0

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() if hasattr(v, 'cuda') else v for key, v in batch.items()}
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        image = Variable(batch['img_feat'], volatile=True)  # [n_batch, imgFeatureSize]


        gtFeatures = image

        qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens, image=image)
        predFeatures = qBot.predictImage()

        # Evaluating round 0 feature regression network
        featLoss = F.mse_loss(predFeatures, gtFeatures)
        featLossAll[0].append(torch.mean(featLoss))
        # Keeping round 0 predictions
        roundwiseFeaturePreds[0].append(predFeatures)
        cur_dialog_hidden = None
        past_dialog_hidden = None
        
        for round in range(numRounds):
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            qBot.observe(
                round, ans=answers[:, round], ansLens=ansLens[:, round])
            cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            if round > 0:
                # calculate diversity losses

                # cos similarity
                cos_similarity_loss += utils.cosinePenalty(cur_dialog_hidden, past_dialog_hidden)

                # huber loss
                huber_loss += utils.huberPenalty(cur_dialog_hidden, past_dialog_hidden, threshold=0.1)

            past_dialog_hidden = cur_dialog_hidden

            predFeatures = qBot.predictImage()
            # Evaluating feature regression network
            
            featLoss = F.mse_loss(predFeatures, gtFeatures)
            featLossAll[round + 1].append(torch.mean(featLoss))
            # Keeping predictions
            roundwiseFeaturePreds[round + 1].append(predFeatures)
        gtImgFeatures.append(gtFeatures)


        end_t = timer()
        delta_t = " Time: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []
    poolSize = len(dataset)

    # Keeping tracking of feature regression loss and CE logprobs 
    featLossAll = [torch.mean(torch.stack(floss)) for floss in featLossAll]
    roundwiseFeatLoss = torch.stack(featLossAll).data.cpu().numpy()
    featLossMean = roundwiseFeatLoss.mean()

    huber_loss =  huber_loss /(numRounds * numBatches)
    cos_similarity_loss = cos_similarity_loss/(numRounds * numBatches)

    if verbose:
        print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        # num_examples x num_examples
        dists = pairwise_distances(predFeatures, gtFeatures)
        
        ranks = []
        for i in range(dists.shape[0]):
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        if verbose:
            print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        rankMetricsRounds.append(rankMetrics)

    rankMetricsRounds[-1]['featLossMean'] = featLossMean
    rankMetricsRounds[-1]['huberLossMean'] = huber_loss.data.cpu().numpy()
    rankMetricsRounds[-1]["cosSimilarityLossMean"] = cos_similarity_loss.data.cpu().numpy()
    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds


def rankQBot(qBot, dataset, split, exampleLimit=None, verbose=True, num_workers=1, beamSize=5, params=None):
    '''
        Evaluates Q-Bot performance on image retrieval when it is shown
        ground truth captions, questions and answers. Q-Bot does not
        generate dialog in this setting - it only encodes ground truth
        captions and dialog in order to perform image retrieval by
        predicting FC-7 image features after each round of dialog.

        Arguments:
            qBot    : Q-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn)

    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.item()])) #.encode('utf-8','ignore')

    # enumerate all gt features and all predicted features
    gtImgFeatures = []
    # caption + dialog rounds
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    logProbsAll = [[] for _ in range(numRounds)]
    featLossAll = [[] for _ in range(numRounds + 1)]
    start_t = timer()

    cos_similarity_loss = 0
    huber_loss = 0

    # entity
    r1_ent = [0] * numRounds
    r5_ent = [0] * numRounds
    r10_ent = [0] * numRounds
    r20_ent = [0] * numRounds
    tot_ent = [0] * numRounds

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() if hasattr(v, 'cuda') else v for key, v in batch.items()}
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        image = Variable(batch['img_feat'], volatile=True)  # [n_batch, imgFeatureSize]
        gtFeatures = image

        if params['use_entity']:
            entity_count = Variable(batch['entity_count'], requires_grad=False)
            entity_length = Variable(batch['entity_length'], requires_grad=False)
            entity_id = Variable(batch['entity_id'], requires_grad=False)
            entity_prob = Variable(batch['entity_prob'], requires_grad=False)
            entity_flg = Variable(batch['entity_flg'], requires_grad=False)
            qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens, entity_count=entity_count, entity_length=entity_length, entity_id=entity_id)
        else:
            qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens)

            

        predFeatures = qBot.predictImage()

        # Evaluating round 0 feature regression network
        featLoss = F.mse_loss(predFeatures, gtFeatures)
        featLossAll[0].append(torch.mean(featLoss))
        # Keeping round 0 predictions
        roundwiseFeaturePreds[0].append(predFeatures)
        cur_dialog_hidden = None
        past_dialog_hidden = None

        n_batch = caption.size()[0]
        question_list = [[] for j in range(n_batch)]
        gt_questions_list = [[] for j in range(n_batch)]
        gt_answer_list = [[] for j in range(n_batch)]
        prior = [[] for j in range(n_batch)]

        for round in range(numRounds):
            questions, quesLens = qBot.forwardDecode(
                inference='greedy', beamSize=beamSize)

            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])

            logProbsCurrent = qBot.forward()
            cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]

            qBot.observe(round, ans=answers[:, round], ansLens=ansLens[:, round])  

            for j in range(n_batch):
                gt_questions_list[j].append(to_str_gt(gtQuestions[j, round]))
                gt_answer_list[j].append(to_str_gt(answers[j, round]))
                question_list[j].append(to_str_pred(questions[j], quesLens[j])[8:])
            
            if params['use_entity']:
                for j in range(n_batch):
                    if entity_flg[j, round]==0:
                        continue

                    top1 = qBot.encoder.prior[j].topk(k=1)[1].tolist()
                    top5 = qBot.encoder.prior[j].topk(k=5)[1].tolist()
                    top10 = qBot.encoder.prior[j].topk(k=10)[1].tolist()
                    top20 = qBot.encoder.prior[j].topk(k=20)[1].tolist()
                    ent_gt = [i for i, x in enumerate(entity_prob[j, round]) if x > 0][0]

                    if ent_gt in top1:
                        r1_ent[round] += 1
                    if ent_gt in top5:
                        r5_ent[round] += 1
                    if ent_gt in top10:
                        r10_ent[round] += 1
                    if ent_gt in top20:
                        r20_ent[round] += 1
                    tot_ent[round] += 1

                    qBot.encoder.entity_cnt[j] += torch.zeros_like(qBot.encoder.entity_cnt[j]).cuda()
                    qBot.encoder.entity_cnt[j, ent_gt] += 1
                    qBot.encoder.entity_mask[j] = qBot.encoder.entity_mask[j] | qBot.encoder.entity_cnt[j].ge(qBot.encoder.frequency_limit)
            
            if round > 0:
                # calculate diversity losses

                # cos similarity
                cos_similarity_loss += utils.cosinePenalty(cur_dialog_hidden, past_dialog_hidden)

                # huber loss
                huber_loss += utils.huberPenalty(cur_dialog_hidden, past_dialog_hidden, threshold=0.1)

            past_dialog_hidden = cur_dialog_hidden

            # Evaluating logProbs for cross entropy
            logProbsAll[round].append(
                utils.maskedNll(logProbsCurrent,
                                gtQuestions[:, round].contiguous()))
            predFeatures = qBot.predictImage()
            # Evaluating feature regression network
            featLoss = F.mse_loss(predFeatures, gtFeatures)
            featLossAll[round + 1].append(torch.mean(featLoss))
            # Keeping predictions
            roundwiseFeaturePreds[round + 1].append(predFeatures)
        gtImgFeatures.append(gtFeatures)

        end_t = timer()
        delta_t = " Time: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    if params['use_entity']:
        print('Entity Accu')
        print('r1:')
        for round in range(10):
            print((round, r1_ent[round]/tot_ent[round]))
        print('full', sum(r1_ent) / sum(tot_ent))
        print('===============================')
        print('r5:')
        for round in range(10):
            print((round, r5_ent[round]/tot_ent[round]))
        print('full', sum(r5_ent) / sum(tot_ent))
        print('===============================')
        print('r10:')
        for round in range(10):
            print((round, r10_ent[round]/tot_ent[round]))
        print('full', sum(r10_ent) / sum(tot_ent))
        print('===============================')
        print('sum of Q:')
        for round in range(10):
            print((round, tot_ent[round]))

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []
    poolSize = len(dataset)

    # Keeping tracking of feature regression loss and CE logprobs
    logProbsAll = [torch.mean(torch.stack(lprobs)) for lprobs in logProbsAll]
    # logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll]
    
    featLossAll = [torch.mean(torch.stack(floss)) for floss in featLossAll]
    #featLossAll = [torch.cat(floss, 0).mean() for floss in featLossAll]
    roundwiseLogProbs = torch.stack(logProbsAll).data.cpu().numpy()
    roundwiseFeatLoss = torch.stack(featLossAll).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()
    featLossMean = roundwiseFeatLoss.mean()

    huber_loss =  huber_loss /(numRounds * numBatches)
    cos_similarity_loss = cos_similarity_loss/(numRounds * numBatches)

    if verbose:
        print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        # num_examples x num_examples
        dists = pairwise_distances(predFeatures, gtFeatures)
        ranks = []
        for i in range(dists.shape[0]):
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        if verbose:
            print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        # rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        if round < len(roundwiseLogProbs):
            rankMetrics['logProbs'] = roundwiseLogProbs[round]
        rankMetricsRounds.append(rankMetrics)

    rankMetricsRounds[-1]['logProbsMean'] = logProbsMean
    rankMetricsRounds[-1]['featLossMean'] = featLossMean
    rankMetricsRounds[-1]['huberLossMean'] = huber_loss.data.cpu().numpy()
    rankMetricsRounds[-1]["cosSimilarityLossMean"] = cos_similarity_loss.data.cpu().numpy()
    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds

def rankQABots(qBot, aBot, dataset, split, exampleLimit=None, beamSize=5, num_workers=1, verbose=True, params=None):
    '''
        Evaluates Q-Bot and A-Bot performance on image retrieval where
        both agents must converse with each other without any ground truth
        dialog. The common caption shown to both agents is not the ground
        truth caption, but is instead a caption generated (pre-computed)
        by a pre-trained captioning model (neuraltalk2).

        Arguments:
            qBot    : Q-Bot
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
            beamSize     : Beam search width for generating utterrances
    '''

    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn)

    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.item()])) #.encode('utf-8','ignore')

    # train questions
    train_questions = set()
    for idx, batch in enumerate(dataloader):
        # append all questions in train in a set to calculate downstream metrics
        gtQuestions = Variable(batch['ques'], requires_grad=False)
        gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
        if gtQuesLens.shape[0] < batchSize:
            break

        # iterate through the batch and add to dictionary
        for j in range(batchSize):
            for rnd in range(numRounds):
                question_str = to_str_pred(gtQuestions[j,rnd,:], gtQuesLens[j,rnd])
                train_questions.add(question_str[8:])
    
    print('train question len:', len(train_questions))

    gtImgFeatures = []
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]


    # dialog quality
    tot_examples = 0
    unique_questions = 0
    unique_questions_list = []
    all_question_list = []
    mutual_overlap_list = []
    ent_1_list = []
    ent_2_list = []
    dist_1_list = []
    dist_2_list = []
    avg_precision_list = []

    bleu_metric = 0
    novel_questions = 0
    oscillating_questions_cnt = 0
    per_round_bleu = np.zeros(numRounds)
    ent_1 = 0
    ent_2 = 0
    wpt = TreebankWordTokenizer()

    mse_criterion = torch.nn.MSELoss(reduce=False)

    start_t = timer()
    for idx, batch in enumerate(dataloader):
        gc.collect()
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}
        else:
            batch = {key: v.contiguous() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}

        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        image = Variable(batch['img_feat'], volatile=True)  # [n_batch, imgFeatureSize]


        gtFeatures = image
        aBot.eval(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)

        qBot.eval(), qBot.reset()
        if params['use_entity']:
            entity_count = Variable(batch['entity_count'], requires_grad=False)
            entity_length = Variable(batch['entity_length'], requires_grad=False)
            entity_id = Variable(batch['entity_id'], requires_grad=False)
            qBot.observe(-1, caption=caption, captionLens=captionLens, image=image, entity_count=entity_count, entity_length=entity_length, entity_id=entity_id)
        
        else:
            qBot.observe(-1, caption=caption, captionLens=captionLens, image=image)

        predFeatures = qBot.predictImage()
        prevGuess = predFeatures
        roundwiseFeaturePreds[0].append(predFeatures)

        n_batch = caption.size()[0]
        question_list = [[] for j in range(n_batch)]
        answer_list = [[] for j in range(n_batch)]
        rwd1_list = [[] for j in range(n_batch)]
        rwd2_list = [[] for j in range(n_batch)]
        gt_questions_list = [[] for j in range(n_batch)]

        # image_id_list = image_id.tolist()
        obs_id_list = [[] for j in range(n_batch)]
        attn_weight_list = [[] for j in range(n_batch)]


        for round in range(numRounds):
            

            questions, quesLens = qBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            
            predFeatures = qBot.predictImage()
            roundwiseFeaturePreds[round + 1].append(predFeatures)


            for j in range(n_batch):
                question_list[j].append(to_str_pred(questions[j], quesLens[j])[8:])
                answer_list[j].append(to_str_pred(answers[j], ansLens[j])[8:])
                gt_questions_list[j].append(to_str_gt(gtQuestions[j, round]))




        gtImgFeatures.append(gtFeatures)

        per_round_bleu_batch = np.zeros((numRounds, n_batch))
        for j in range(n_batch):
            # calculate bleu scores for each question str, with other questions as references to calculate
            # mutual overlap
            # also calculate round by round bleu score
            unigrams = []
            bigrams = []
            avg_bleu_score = 0
            for rnd in range(numRounds):
                # Novel sentences metric
                cur_ques = question_list[j][rnd]
                gt_ques = gt_questions_list[j][rnd]
                if cur_ques not in train_questions:
                    novel_questions += 1

                # question oscillation metrics
                if rnd >= 2:
                    if cur_ques == question_list[j][rnd-2]:
                        oscillating_questions_cnt += 1

                # bleu/mutual overlap metric
                references = []
                for k in range(numRounds):
                    if rnd != k:
                        references.append(wpt.tokenize(question_list[j][k]))

                avg_bleu_score += sentence_bleu(references,wpt.tokenize(cur_ques))
                per_round_bleu_batch[rnd][j] = sentence_bleu([wpt.tokenize(gt_ques)],
                                                             wpt.tokenize(cur_ques))
                unigrams.extend(list(ngrams(wpt.tokenize(cur_ques),1)))
                bigrams.extend(list(ngrams(wpt.tokenize(cur_ques),2)))

            avg_bleu_score /=  float(numRounds)
            mutual_overlap_list.append(avg_bleu_score)
            bleu_metric += avg_bleu_score
            tot_tokens = len(unigrams)

            unigram_ctr = Counter(unigrams)
            bigram_ctr = Counter(bigrams)
            cur_ent_1 = get_entropy_ctr(unigram_ctr)
            ent_1 += cur_ent_1
            ent_1_list.append(cur_ent_1)
            cur_ent_2 = get_entropy_ctr(bigram_ctr)
            ent_2 += cur_ent_2
            ent_2_list.append(cur_ent_2)

            dist_1 = len(unigram_ctr.keys())/float(tot_tokens)
            dist_2 = len(bigram_ctr.keys())/float(tot_tokens)

            dist_1_list.append(dist_1)
            dist_2_list.append(dist_2)

            cur_unique_ques = len(set(question_list[j]))
            unique_questions += cur_unique_ques
            unique_questions_list.append(cur_unique_ques)
            all_question_list += question_list[j]
        
        tot_examples += n_batch
        
        # if idx%params['print_every']==0 and verbose:
        #     cap = to_str_gt(caption[0])[8:-6]
        #     print('image_id: ', image_id_list[0])
        #     print('C: ', cap)
        #     for round in range(numRounds):
        #         ques = question_list[0][round]
        #         ans = answer_list[0][round]
                
        #         print('Q: ', ques.encode('ascii', 'ignore').decode('ascii'), ' ', 'A: ', ans.encode('ascii', 'ignore').decode('ascii'))
            

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []

    print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        dists = pairwise_distances(predFeatures, gtFeatures)
        # num_examples x num_examples
        ranks = []
        for i in range(dists.shape[0]):
            # Computing rank of i-th prediction vs all images in split
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        assert len(ranks) == len(dataset)
        poolSize = len(dataset)
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetricsRounds.append(rankMetrics)

    print('top 20 sentences')
    cnt = Counter(all_question_list).most_common(20)
    for k, v in cnt:
        print(k, v)

    bleu_metric /= float(tot_examples)
    ent_1 /= float(tot_examples)
    ent_2 /= float(tot_examples)
    per_round_bleu = per_round_bleu / float(tot_examples)

    print("tot unique questions: ", unique_questions)
    print("tot examples: ", tot_examples)
    print("avg unique questions per example: ", float(unique_questions) / tot_examples)
    print("Mutual Overlap (Bleu Metric): ", bleu_metric)
    print("tot novel questions: ", novel_questions)
    tot_questions = tot_examples * numRounds
    print("tot questions: ", tot_questions)
    print("avg novel questions: ", float(novel_questions)/float(tot_questions))

    print("avg oscillating questions count", float(oscillating_questions_cnt)/tot_questions)
    print("osciallation questions count", oscillating_questions_cnt)

    rankMetricsRounds[-1]["tot_unique_questions"] = unique_questions
    rankMetricsRounds[-1]["tot_examples"] = tot_examples
    rankMetricsRounds[-1]["mean_unique_questions"] = int((float(unique_questions) / tot_examples) * 100)/100.0

    rankMetricsRounds[-1]["mutual_overlap_score"] = bleu_metric
    rankMetricsRounds[-1]["tot_novel_questions"] = novel_questions
    rankMetricsRounds[-1]["avg_novel_questions"] = float(novel_questions)/float(tot_questions)
    rankMetricsRounds[-1]["tot_questions"] = tot_questions

    rankMetricsRounds[-1]["average_precision"] = np.mean(per_round_bleu)
    rankMetricsRounds[-1]["per_round_precision"] = per_round_bleu.tolist()
    rankMetricsRounds[-1]["ent_1"] = ent_1
    rankMetricsRounds[-1]["ent_2"] = ent_2
    rankMetricsRounds[-1]["dist_1"] = np.mean(dist_1_list)
    rankMetricsRounds[-1]["dist_2"] = np.mean(dist_2_list)

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds


def get_entropy_ctr(ctr):

    values = list(ctr.values())
    sum_values = float(sum(values))
    probs = [x/sum_values for x in values]
    return entropy(probs)



# use referee Q-Bot for evaluation
def rankQABots_with_guesser(qBot, guesser, aBot, dataset, split, exampleLimit=None, beamSize=5, num_workers=1, verbose=True, params=None):
    '''
        Evaluates Q-Bot and A-Bot performance on image retrieval where
        both agents must converse with each other without any ground truth
        dialog. The common caption shown to both agents is not the ground
        truth caption, but is instead a caption generated (pre-computed)
        by a pre-trained captioning model (neuraltalk2).

        Arguments:
            qBot    : Q-Bot
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
            beamSize     : Beam search width for generating utterrances
    '''

    qBot.eval()
    aBot.eval()
    guesser.eval()

    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn)

    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.item()])) #.encode('utf-8','ignore')

    # train questions
    train_questions = set()
    for idx, batch in enumerate(dataloader):
        # append all questions in train in a set to calculate downstream metrics
        gtQuestions = Variable(batch['ques'], requires_grad=False)
        gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
        if gtQuesLens.shape[0] < batchSize:
            break

        # iterate through the batch and add to dictionary
        for j in range(batchSize):
            for rnd in range(min(10, numRounds)):
                question_str = to_str_pred(gtQuestions[j,rnd,:], gtQuesLens[j,rnd])
                train_questions.add(question_str[8:])
    
    print('train question len:', len(train_questions))

    gtImgFeatures = []
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]


    # dialog quality
    tot_examples = 0
    unique_questions = 0
    unique_questions_list = []
    all_question_list = []
    mutual_overlap_list = []
    ent_1_list = []
    ent_2_list = []
    dist_1_list = []
    dist_2_list = []
    avg_precision_list = []

    bleu_metric = 0
    novel_questions = 0
    oscillating_questions_cnt = 0
    per_round_bleu = np.zeros(numRounds)
    ent_1 = 0
    ent_2 = 0
    wpt = TreebankWordTokenizer()

    mse_criterion = torch.nn.MSELoss(reduce=False)
    
    freq_questions = ['can you see trees ?', 'can you see the sky ?', 'can you see the floor ?', 'can you see buildings ?', 'can you see any trees ?', 'can you see the walls ?', 'is the photo in color ?', 'do you see any trees ?', 'can you see any buildings ?', 'what color is his hair ?', 'is it sunny ?', 'is the photo close up ?', 'is it day or night ?', 'are there any people ?', 'what color is the table ?', 'are there any trees ?', 'can you see grass ?', 'what color is her hair ?', 'is he wearing glasses ?', 'do you see any buildings ?']
    
    dialog_list = []

    start_t = timer()

    prob = [[0]*(params['n_entity']+1) for i in range(numRounds)]

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}
        else:
            batch = {key: v.contiguous() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}

        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        image = Variable(batch['img_feat'], volatile=True)  # [n_batch, imgFeatureSize]
        image_id = Variable(batch['image_id'], volatile=True)

        gtFeatures = image

        aBot.eval(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        

        if params['use_entity']:
            entity_count = Variable(batch['entity_count'], requires_grad=False)
            entity_length = Variable(batch['entity_length'], requires_grad=False)
            entity_id = Variable(batch['entity_id'], requires_grad=False)
            qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens, entity_count=entity_count, entity_length=entity_length, entity_id=entity_id)
        else:
            qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens)
        
        guesser.reset()
        guesser.observe(-1, caption=caption, captionLens=captionLens)
        

        predFeatures = guesser.predictImage()
        prevGuess = predFeatures
        roundwiseFeaturePreds[0].append(predFeatures)

        n_batch = caption.size()[0]
        question_list = [[] for j in range(n_batch)]
        answer_list = [[] for j in range(n_batch)]
        gt_questions_list = [[] for j in range(n_batch)]
        reward_list = [[] for j in range(n_batch)]


        # use for candidate image
        image_id_list = image_id.tolist()
        obs_id_list = [[] for j in range(n_batch)]
        attn_weight_list = [[] for j in range(n_batch)]

        # entity information
        prior = [[] for j in range(n_batch)]
        sample_list = [[] for j in range(n_batch)]

        for round in range(numRounds):

            questions, quesLens = qBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            qBot.observe(round, ques=questions, quesLens=quesLens)
            guesser.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)


            answers, ansLens = aBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            qBot.observe(round, ans=answers, ansLens=ansLens)
            guesser.observe(round, ans=answers, ansLens=ansLens)
            aBot.observe(round, ans=answers, ansLens=ansLens)            

            predFeatures = guesser.predictImage()
            reward = (mse_criterion(image, prevGuess) - mse_criterion(image, predFeatures)).mean(1)  # (n_batch)
            prevGuess = predFeatures
            roundwiseFeaturePreds[round + 1].append(predFeatures)

            # use for entity information
            if params['use_entity']:
                for j in range(n_batch):
                    sample_list[j] += [[i for i, x in enumerate(qBot.encoder.sample_ind[j].tolist()) if x>0]]
                    prior[j] += [qBot.encoder.prior[j].tolist()]
                
                for j in range(n_batch):
                    for i in range(params['n_entity']+1):
                        prob[round][i] += prior[j][-1][i]


            for j in range(n_batch):
                question_list[j].append(to_str_pred(questions[j], quesLens[j])[8:])
                answer_list[j].append(to_str_pred(answers[j], ansLens[j])[8:])
                if round < min(10, numRounds):
                    gt_questions_list[j].append(to_str_gt(gtQuestions[j, round]))
                reward_list[j].append(reward[j].item())



        gtImgFeatures.append(gtFeatures)

        per_round_bleu_batch = np.zeros((numRounds, n_batch))
        for j in range(n_batch):
            # calculate bleu scores for each question str, with other questions as references to calculate
            # mutual overlap
            # also calculate round by round bleu score
            unigrams = []
            bigrams = []
            avg_bleu_score = 0
            for rnd in range(numRounds):
                # Novel sentences metric
                cur_ques = question_list[j][rnd]
                # gt_ques = gt_questions_list[j][rnd]
                if cur_ques not in train_questions:
                    novel_questions += 1

                # question oscillation metrics
                if rnd >= 2:
                    if cur_ques == question_list[j][rnd-2]:
                        oscillating_questions_cnt += 1

                # bleu/mutual overlap metric
                references = []
                for k in range(numRounds):
                    if rnd != k:
                        references.append(wpt.tokenize(question_list[j][k]))

                avg_bleu_score += sentence_bleu(references,wpt.tokenize(cur_ques))
                # per_round_bleu_batch[rnd][j] = sentence_bleu([wpt.tokenize(gt_ques)],
                #                                              wpt.tokenize(cur_ques))
                unigrams.extend(list(ngrams(wpt.tokenize(cur_ques),1)))
                bigrams.extend(list(ngrams(wpt.tokenize(cur_ques),2)))

            avg_bleu_score /=  float(numRounds)
            mutual_overlap_list.append(avg_bleu_score)
            bleu_metric += avg_bleu_score
            tot_tokens = len(unigrams)

            unigram_ctr = Counter(unigrams)
            bigram_ctr = Counter(bigrams)
            cur_ent_1 = get_entropy_ctr(unigram_ctr)
            ent_1 += cur_ent_1
            ent_1_list.append(cur_ent_1)
            cur_ent_2 = get_entropy_ctr(bigram_ctr)
            ent_2 += cur_ent_2
            ent_2_list.append(cur_ent_2)

            dist_1 = len(unigram_ctr.keys())/float(tot_tokens)
            dist_2 = len(bigram_ctr.keys())/float(tot_tokens)

            dist_1_list.append(dist_1)
            dist_2_list.append(dist_2)

            cur_unique_ques = len(set(question_list[j]))
            unique_questions += cur_unique_ques
            unique_questions_list.append(cur_unique_ques)
            all_question_list += question_list[j]
        
        tot_examples += n_batch
        

        for bn in range(n_batch):
            dialog_meta = {}
            dialog_meta['image_id'] = image_id_list[bn]
            dialog_meta['cap'] = to_str_gt(caption[bn])[8:-6]
            dialog = []
            for round in range(numRounds):
                ques = question_list[bn][round]
                ans = answer_list[bn][round]
                rwd = reward_list[bn][round]
                
                dialog += [{'ques':ques, 'ans':ans, 'rwd':rwd}]
            dialog_meta['dialog'] = dialog

            dialog_list += [dialog_meta]

        # print dialog
        if idx%params['print_every']==0 and verbose:
            cap = to_str_gt(caption[0])[8:-6]
            print('image_id: ', image_id_list[0])
            print('C: ', cap)

            if params['use_entity']:
                entity_list = entity_id[0, :entity_count[0].item()] # (n_ent, max_ent_len)
                entity_list = [to_str_gt(entity_list[i]) for i in range(entity_count[0].item())]
                print(','.join(entity_list))

            for round in range(numRounds):
                ques = question_list[0][round]
                ans = answer_list[0][round]
                rwd = reward_list[0][round]
                print('Q: ', ques.encode('ascii', 'ignore').decode('ascii'), ' ', 'A: ', ans.encode('ascii', 'ignore').decode('ascii'), ' rwd: %.3g' %rwd)
                

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []

    print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        dists = pairwise_distances(predFeatures, gtFeatures)
        # num_examples x num_examples
        ranks = []
        for i in range(dists.shape[0]):
            # Computing rank of i-th prediction vs all images in split
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        assert len(ranks) == len(dataset)
        poolSize = len(dataset)
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetricsRounds.append(rankMetrics)    

    print('top 20 sentences')
    cnt = Counter(all_question_list).most_common(20)
    for k, v in cnt:
        print(k, v)


    bleu_metric /= float(tot_examples)
    ent_1 /= float(tot_examples)
    ent_2 /= float(tot_examples)
    per_round_bleu = per_round_bleu / float(tot_examples)

    print("tot unique questions: ", unique_questions)
    print("tot examples: ", tot_examples)
    print("avg unique questions per example: ", float(unique_questions) / tot_examples)
    print("Mutual Overlap (Bleu Metric): ", bleu_metric)
    print("tot novel questions: ", novel_questions)
    tot_questions = tot_examples * numRounds
    print("tot questions: ", tot_questions)
    print("avg novel questions: ", float(novel_questions)/float(tot_questions))

    print("avg oscillating questions count", float(oscillating_questions_cnt)/tot_questions)
    print("osciallation questions count", oscillating_questions_cnt)

    rankMetricsRounds[-1]["tot_unique_questions"] = unique_questions
    rankMetricsRounds[-1]["tot_examples"] = tot_examples
    rankMetricsRounds[-1]["mean_unique_questions"] = int((float(unique_questions) / tot_examples) * 100)/100.0

    rankMetricsRounds[-1]["mutual_overlap_score"] = bleu_metric
    rankMetricsRounds[-1]["tot_novel_questions"] = novel_questions
    rankMetricsRounds[-1]["avg_novel_questions"] = float(novel_questions)/float(tot_questions)
    rankMetricsRounds[-1]["tot_questions"] = tot_questions

    rankMetricsRounds[-1]["average_precision"] = np.mean(per_round_bleu)
    rankMetricsRounds[-1]["per_round_precision"] = per_round_bleu.tolist()
    rankMetricsRounds[-1]["ent_1"] = ent_1
    rankMetricsRounds[-1]["ent_2"] = ent_2
    rankMetricsRounds[-1]["dist_1"] = np.mean(dist_1_list)
    rankMetricsRounds[-1]["dist_2"] = np.mean(dist_2_list)



    for round in range(numRounds):
        for i in range(params['n_entity'] + 1):
            prob[round][i] /= tot_examples
    
    # for round in range(numRounds):
    #     print(f'round[{round}] distribution: ', prob[round])

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds
