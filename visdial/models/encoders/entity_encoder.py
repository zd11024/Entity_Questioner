import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import utilities as utils


def masked_softmax(vec, mask, dim=1, log_softmax=False):
    out = F.softmax(vec.masked_fill((mask).bool(), float('-inf')), dim=dim)
    if log_softmax:
        out = F.normalize(out+1e-10, p=1)
        out = torch.log(out)
    return out

class Att(nn.Module):
    def __init__(self, d_x, d_y, nhid, dropout=0., temperature=1.):
        super(Att, self).__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.nhid = nhid
        self.dropout = dropout
        self.temperature = temperature
        self.Wk_1 = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_x, self.nhid)
            )
        self.Wq_1 = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_y, self.nhid)
            )
        self.Wa_1 = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.nhid, 1)
            )
    
    def forward(self, k, q, mask, log_softmax=False):
        """
        k: [n_batch, n_seq, d_emb]
        q: [n_batch, d_emb]
        """
        k_emb = self.Wk_1(k)
        q_emb = self.Wq_1(q).unsqueeze(1)
        attn_weight = self.Wa_1(F.tanh(k_emb + q_emb)).squeeze(-1) / self.temperature
        out = masked_softmax(attn_weight, mask, log_softmax=log_softmax)
        return out

def cond_gumbel_softmax(logits, is_training, topk=1, temperature=1.0):

    y = F.gumbel_softmax(logits, tau=temperature, dim=-1) if is_training else F.softmax(logits, dim=-1)

    y_detach = y.detach()
    _, y_index = y_detach.topk(topk, dim=-1)  # (n_batch, n_cand, topk)
    y_hard = torch.zeros_like(logits).scatter_(-1, y_index, 1)

    out = y_hard - y.detach() + y if is_training else y_hard
    return out
    

class Encoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 useIm,
                 imgEmbedSize,
                 imgFeatureSize,
                 numRounds,
                 isAnswerer,
                 dropout=0,
                 startToken=None,
                 endToken=None,
                 frequency_limit=1,
                 **kwargs):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken
        self.nhid = rnnHiddenSize
        self.tot1 = 0
        self.tot2 = 0
        self.frequency_limit = frequency_limit

        # modules
        self.wordEmbed = nn.Embedding(
            self.vocabSize, self.embedSize, padding_idx=0)

        # question encoder
        # image fuses early with words
        if self.isAnswerer:
            if self.useIm == 'early':
                quesInputSize = self.embedSize + self.imgEmbedSize
                dialogInputSize = 2 * self.rnnHiddenSize
                self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
                self.imgEmbedDropout = nn.Dropout(self.dropout)
            elif self.useIm == 'late':
                quesInputSize = self.embedSize
                dialogInputSize = 2 * self.rnnHiddenSize + self.imgEmbedSize
                self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
                self.imgEmbedDropout = nn.Dropout(self.dropout)
            else:
                quesInputSize = self.embedSize
                dialogInputSize = self.rnnHiddenSize
        else:
            # use candicate image feature
            dialogInputSize = self.rnnHiddenSize

        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0)

        # question encoder
        self.quesRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0)

        # entity encoder
        self.entityRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            1,
            batch_first=True,
            dropout=0)
        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)


        # Setup feature regressor
        if not self.isAnswerer:
            self.featureNet = nn.Linear(self.rnnHiddenSize,
                                        self.imgFeatureSize)
            self.featureNetInputDropout = nn.Dropout(self.dropout)
            
            self.att_prior = Att(self.rnnHiddenSize, self.rnnHiddenSize*3, self.rnnHiddenSize)



        # initialization
        for module in self.modules():
            name = type(module).__name__
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)


    def reset(self):
        self.alpha = []
        # batchSize is inferred from input
        self.batchSize = 0
        if self.training:
            self.tot1 += 1
            self.tot2 = 0
        else:
            self.tot1 = 0
            self.tot2 += 1
        if self.tot1%1000==0 and self.training:
            self.verbose=True
        elif self.tot2%100==0 and not self.training:
            self.verbose=True
        else:
            self.verbose = False

        self.captionTokens = None
        self.captionEmbed = None
        self.captionLens = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []
        self.encoderStates = {}  # round to encoderStates

        # entity embeddings
        self.enitity_count = None
        self.entity_length = None
        self.entity_id = None
        self.entity_embed = None
        self.entity_embed_lstm = None



    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def observe(self,
                round,
                image=None,
                caption=None,
                ques=None,
                ans=None,
                captionLens=None,
                quesLens=None,
                ansLens=None,
                entity_count=None,
                entity_length=None,
                entity_id=None,
                entity_prob=None):
        '''
        Store dialog input to internal model storage
        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''

        if caption is not None:
            assert round == -1
            assert captionLens is not None, "Caption lengths required!"
            caption, captionLens = self.processSequence(caption, captionLens)
            self.captionTokens = caption
            self.captionLens = captionLens
            self.batchSize = len(self.captionTokens)
        if ques is not None:
            assert round == len(self.questionEmbeds)
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)
        
        if entity_count is not None:
            assert round==-1
            self.entity_count=entity_count
            self.entity_length=entity_length
            self.entity_id=entity_id

            self.entity_embed = self.wordEmbed(self.entity_id)  # (n_batch, n_ent, ent_len, d_emb)

            # embed entity
            n_batch, n_ent, ent_len, _ = self.entity_embed.size()
            seq = self.entity_embed.view(n_batch * n_ent, ent_len, self.embedSize)
            seqLens = self.entity_length.view(n_batch * n_ent)
            entity_embed_lstm, states = utils.dynamicRNN(
                self.entityRNN, seq, seqLens, returnStates=True)
            self.entity_embed_lstm = entity_embed_lstm.view(n_batch, n_ent, self.rnnHiddenSize)

            indices = torch.arange(n_ent).unsqueeze(0).expand(n_batch, n_ent).contiguous().cuda()
            self.entity_mask = indices >= entity_count.view(-1, 1)  # init candicate mask
            self.entity_cnt = torch.zeros_like(self.entity_mask).float()
            self.entity_prob = entity_prob  # (n_batch, n_round, n_ent)


    def predictImage(self):
        """
        transform history hidden to image feature.
        """
        encState = self.forward(predict=True)
        dialogHidden = self.dialogHiddens[-1][0]
        return self.featureNet(self.featureNetInputDropout(dialogHidden))

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''

        # Embed caption, occurs once per dialog
        if self.captionEmbed is None:
            self.captionEmbed = self.wordEmbed(self.captionTokens)

        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(
                self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        '''Embed facts i.e. caption and round 0 or question-answer pair otherwise'''
        # Caption
        if factIdx == 0:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(
                self.factRNN, seq, seqLens, returnStates=True)

        # QA pairs
        elif factIdx > 0:
            quesTokens, quesLens = \
                self.questionTokens[factIdx - 1], self.questionLens[factIdx - 1]
            ansTokens, ansLens = \
                self.answerTokens[factIdx - 1], self.answerLengths[factIdx - 1]

            qaTokens = utils.concatPaddedSequences(
                quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(
                self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
            if self.useIm == 'late':
                currIns.append(self.imageEmbed)
        
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)


    # mode include sl, rl, test
    def forward(self, predict=False, mode='sl'):
        ''' 
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hi-dden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''
        # Lazily embed input Image, Captions, Questions and Answers
        self.embedInputDialog()

        if self.isAnswerer:
            # For A-Bot, current round is the number of facts present,
            # which is number of questions observed - 1 (as opposed
            # to len(self.answerEmbeds), which may be inaccurate as
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)
        

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # # Embed any un-embedded questions (A-Bot only)
        # if self.isAnswerer:
        #     while len(self.questionRNNStates) <= round:
        #         qIdx = len(self.questionRNNStates)
        #         self.embedQuestion(qIdx)

        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]

        if predict:
            return dialogHidden
        elif round in self.encoderStates:  # memoization
            return self.encoderStates[round]

        # calculate encoder states
        if mode=='sl':
            # for calculate posterior
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx)
        
        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)t
              Layer 1 : DialogRNN hidden state (dialogRNN)
        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN)
        '''
        if self.isAnswerer:
            quesRNNstates = self.questionRNNStates[-1][1]  # Latest quesRNN states
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            n_batch = dialogHidden.size(0)

            factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
            captRNNstates = self.factEmbeds[0][1]  # [(2, n_batch, nhid), (2, n_batch, nhid)]
            
            H_link = factRNNstates[0][0]
            C_link = factRNNstates[1]

            entity_emb = self.HCIAE(round, mode=mode)
            H_link2 = dialogHidden
            # H_link2 = self.fc1(torch.cat([F.dropout(dialogHidden, p=self.hidden_dropout, training=self.training), entity_emb], dim=1))

            H_link = torch.cat([H_link.unsqueeze(0), H_link2.unsqueeze(0)], 0)
            self.hidden = H_link2
            self.entity_emb = entity_emb
        
        self.encoderStates[round] = (H_link, C_link)
        return H_link, C_link


    def HCIAE(self, round, mode):
        fact = self.factEmbeds[-1][0]
        hist = self.dialogHiddens[-1][0]
        capt = self.factEmbeds[0][0]
        assert len(self.dialogHiddens)==round+1
        assert len(self.factEmbeds)==round+1

        if self.training and mode=='sl':
            prior_logits = self.att_prior(self.entity_embed_lstm, torch.cat([capt, fact, hist], dim=1), self.entity_mask, log_softmax=True)  # log_softmax
            prior = torch.exp(prior_logits)

            posterior = self.entity_prob[:, round]

            sample_ind = torch.multinomial(posterior, num_samples=1, replacement=False)  # (n_batch, 1)
            sample_ind = torch.zeros_like(prior).scatter_(-1, sample_ind, 1)  # (n_batch, n_ent)

            out = (sample_ind.unsqueeze(-1) * self.entity_embed_lstm).sum(1)
            self.sample_ind = sample_ind
            self.prior_logits = prior_logits
            self.posterior = posterior

        else:
            prior_logits = self.att_prior(self.entity_embed_lstm, torch.cat([capt, fact, hist], dim=1), self.entity_mask, log_softmax=True)  # log_softmax
            prior = torch.exp(prior_logits)

            assert mode in ['rl', 'test']
            if mode=='rl':
                sample_ind = torch.multinomial(prior, num_samples=1, replacement=False)
                
                sample_ind = torch.zeros_like(prior).scatter_(-1, sample_ind, 1)
                out = (sample_ind.unsqueeze(-1) * self.entity_embed_lstm).sum(1)
            elif mode=='test':
                sample_ind = torch.multinomial(prior, num_samples=1, replacement=False)
                # sample_ind = prior.max(1, keepdim=True)[1]  # (n_batch, 1) 
                sample_ind = torch.zeros_like(prior).scatter_(-1, sample_ind, 1)
                out = (sample_ind.unsqueeze(-1) * self.entity_embed_lstm).sum(1)
                self.entity_cnt += sample_ind
                self.entity_mask = self.entity_mask | self.entity_cnt.ge(self.frequency_limit)

            self.prior = prior
            self.sample_ind = sample_ind

        return out




if __name__ == '__main__':
    n_batch = 20
    nhid = 16
    n_cand = 10
    pps = 36
    d_img = 48
    n_ent=36
    m = Encoder(vocabSize=20,
                 embedSize=10,
                 rnnHiddenSize=nhid,
                 numLayers=2,
                 useIm=False,
                 imgEmbedSize=nhid,
                 imgFeatureSize=d_img,
                 numRounds=10,
                 isAnswerer=False,
                 dropout=0,
                 startToken=None,
                 endToken=None)
    
    m.reset()
    m.train()
    m.factEmbeds = [(torch.zeros(n_batch, nhid), 0)]
    m.questionRNNStates = [(torch.zeros(n_batch, nhid), 0)]
    m.dialogHiddens = [(torch.zeros(n_batch, nhid), 0)]
    
    m.entity_embed_lstm = torch.ones(n_batch, n_ent, nhid)
    m.entity_mask = torch.zeros(n_batch, n_ent).bool()
    m.entity_cnt = torch.zeros(n_batch, n_ent)
    m.entity_prob = F.normalize(torch.ones(n_batch, 10, n_ent), p=1)
    m.batchSize = n_batch
    m.verbose = True
    out = m.HCIAE(0)
    print(out.size())