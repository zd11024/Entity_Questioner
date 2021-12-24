import importlib
import torch
import torch.nn as nn
from visdial.models.agent import Agent
import visdial.models.encoders.entity_encoder as hre_enc
import visdial.models.decoders.entity_decoder as gen_dec
from utils import utilities as utils

class Questioner(Agent):
    def __init__(self, encoderParam, decoderParam, imgFeatureSize=0,
                 verbose=1,multiGPU=False, params=None):
        '''
            Q-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (question).
        '''
        super(Questioner, self).__init__()
        # self.encType = encoderParam['type']
        # self.decType = decoderParam['type']
        self.encType = 'entity_encoder'
        self.decType = 'entity_decoder'
        self.dropout = encoderParam['dropout']
        self.rnnHiddenSize = encoderParam['rnnHiddenSize']
        self.imgFeatureSize = imgFeatureSize
        self.multiGPU = multiGPU
        self.params = params
        encoderParam = encoderParam.copy()
        encoderParam['isAnswerer'] = False

        # Encoderm
        if verbose:
            print('Encoder: ' + self.encType)
            print('Encoder Param: ', encoderParam)
            print('Decoder: ' + self.decType)
            print('Decoder Param: ', decoderParam)
        self.encoder = hre_enc.Encoder(**encoderParam)
        if multiGPU:
            self.encoder = nn.DataParallel(self.encoder)

        # Decoder
        self.decoder = gen_dec.Decoder(**decoderParam)
        if multiGPU:
            self.decoder = nn.DataParallel(self.decoder)

        # Share word embedding parameters between encoder and decoder
        if multiGPU:
            self.decoder.module.wordEmbed = self.encoder.module.wordEmbed
        else:
            self.decoder.wordEmbed = self.encoder.wordEmbed

        # Initialize weights
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.reset()

    def reset(self):
        '''Delete dialog history.'''
        self.questions = []
        if self.multiGPU:
            self.encoder.module.reset()
        else:
            self.encoder.reset()

    def freezeFeatNet(self):
        nets = [self.encoder.featureNet]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = False

    def observe(self, round, ques=None, **kwargs):
        '''
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        '''
        # assert 'image' not in kwargs, "Q-Bot does not see image"
        if ques is not None:
            assert round == len(self.questions), \
                "Round number does not match number of questions observed"
            self.questions.append(ques)

        if self.multiGPU:
            self.encoder.observe(round, ques=ques, **kwargs)
        else:
            self.encoder.observe(round, ques=ques, **kwargs)

    def forward(self):
        '''
        Forward pass the last observed question to compute its log
        likelihood under the current decoder RNN state.
        '''
        encStates = self.encoder(mode='sl')
        if len(self.questions) == 0:
            raise Exception('Must provide question if not sampling one.')
        decIn = self.questions[-1]

        if self.decType=='gen':
            logProbs = self.decoder(encStates, inputSeq=decIn)
        else:
            logProbs = self.decoder(encStates, inputSeq=decIn, entity_emb=self.encoder.entity_emb)
        return logProbs

    # mode include rl and test
    def forwardDecode(self, inference='sample', futureReward=False, beamSize=1, maxSeqLen=20,run_mcts=False, mode='test'):
        '''
        Decode a sequence (question) using either sampling or greedy inference.
        A question is decoded given current state (dialog history). This can
        be called at round 0 after the caption is observed, and at end of every
        round (after a response from A-Bot is observed).

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        '''
        encStates = self.encoder(mode=mode)
        if self.decType=='gen':
            questions, quesLens = self.decoder.forwardDecode(
                encStates,
                maxSeqLen=maxSeqLen,
                inference=inference,
                futureReward=futureReward,
                beamSize=beamSize, 
                run_mcts=run_mcts)
        else:
            questions, quesLens = self.decoder.forwardDecode(
                encStates,
                entity_emb=self.encoder.entity_emb,
                maxSeqLen=maxSeqLen,
                inference=inference,
                futureReward=futureReward,
                beamSize=beamSize, run_mcts=run_mcts)
        return questions, quesLens

    def predictImage(self):
        '''
        Predict/guess an fc7 vector given the current conversation history. This can
        be called at round 0 after the caption is observed, and at end of every round
        (after a response from A-Bot is observed).
        '''
        return self.encoder.predictImage()

    def reinforce(self, reward, futureReward=False, mcts=False):
        # Propogate reinforce function call to decoder
        return self.decoder.reinforce(reward, futureReward=futureReward, mcts=mcts)
