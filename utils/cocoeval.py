from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import os, cPickle

class COCOScorer(object):
    def __init__(self):
        print 'init COCO-EVAL scorer'
            
    def score(self, GT, RES, IDs):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print "%s: %0.3f"%(method, score)
                
        for metric, score in self.eval.items():
            print '%s: %.3f'%(metric, score)
        return self.eval
    
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

            
def load_pkl(path):    
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def score(ref, sample):
    # ref and sample are both dict
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        print 'computing %s score with COCO-EVAL...'%(scorer.method())
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def test_cocoscorer():
    """
    gts = {
        '184321':[
        {u'caption': u'A train traveling down tracks next to lights.'},
        {u'caption': u'A train coming down the tracks arriving at a station.'}],
        '81922': [
        {u'caption': u'A large jetliner flying over a traffic filled street.'},
        {u'caption': u'The plane is flying over top of the cars'}]
        }
    samples = {
        '184321': [{u'caption': u'train traveling down a track in front of a road'}],
        '81922': [{u'caption': u'plane is flying through the sky'}],
        }


    IDs = ['184321', '81922']
    """
    gts = {
        184321:[
        {'caption': 'A train traveling down tracks next to lights.'},
        {'caption': 'A train coming down the tracks arriving at a station.'}],
        81922: [
        {'caption': 'A large jetliner flying over a traffic filled street.'},
        {'caption': 'The plane is flying over top of the cars'}]
        }
    samples = {
        184321: [{'caption': 'train traveling down a track in front of a road'}],
        81922: [{'caption': 'plane is flying through the sky'}],
        }

    IDs = [184321, 81922]

    scorer = COCOScorer()
    scorer.score(gts, samples, IDs)
    
if __name__ == '__main__':
    test_cocoscorer()
