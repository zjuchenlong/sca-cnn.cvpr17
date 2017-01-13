import os
import numpy

from cocoeval import COCOScorer
import sca_resnet_branch2b

MAXLEN = 50

def have_dir(save_result_dir):
  if os.path.exists(save_result_dir):
    print('directory %s already exists' %save_result_dir)
  else:
    os.makedirs(save_result_dir)

def generate_sample(valid, test, word_idict, f_init, f_next, options, beam, save_result_dir):
  
  def _seq2words(caps):
    capsw = []
    for cc in caps:    
      ww = []
      for w in cc:
        if w == 0:
          break
        ww.append(word_idict[1] if w not in word_idict else word_idict[w])
      capsw.append(' '.join(ww))
    return capsw
 
  def sample(split):
   
    samples = []
    assert split[1].shape[0] == split[2].shape[0], "two layer feature should have the same samples"
    for i in range(split[1].shape[0]):
      ctx_res5b = split[1][i]
      ctx_res5c_branch2b = split[2][i]
      sample, sample_score = sca_resnet_branch2b.gen_sample(None, f_init, f_next, ctx_res5b, ctx_res5c_branch2b, options, None, beam, maxlen=MAXLEN)
      sidx = numpy.argmin(sample_score)
      sample = sample[sidx]
      samples.append(sample)
      print("genearate the captions of sample %d" % i)
    
    samples = _seq2words(samples)
    return samples
  
  have_dir(save_result_dir) 
  samples_valid = None
  samples_test = None

  if valid:
    print 'Valid Set... \n',
    samples_valid = sample(valid)
    with open(save_result_dir + '/valid_samples.txt', 'w') as f:
      print >> f, '\n'.join(samples_valid)
 
  if test:
    print 'Test Set... \n',
    samples_test = sample(test)
    with open(save_result_dir + '/test_samples.txt', 'w') as f:
      print >> f, '\n'.join(samples_test)

  return samples_valid, samples_test

def make_template(samples, dataset):
  num_hypo = len(samples)
  num_gts = len(dataset[0])
  
  samples_ids = []
  gts_samples = {}
  hypo_samples = {}

  for i in range(num_hypo):
    samples_ids.append(i)
    hypo_samples[i] = [{'caption':samples[i]}]

  for i in range(num_gts):
    gts = dataset[0][i]
    if gts[1] not in gts_samples:
      gts_samples[gts[1]] = [{'caption': gts[0]}]
    else:
      gts_samples[gts[1]].append({'caption':gts[0]})
    
  return gts_samples, hypo_samples, samples_ids

def score_with_cocoeval(samples_valid, samples_test, valid, test):
  scorer = COCOScorer()

  if samples_valid:

    gts_valid, hypo_valid, valid_ids = make_template(samples_valid, valid)
    print 'compute validation set score:'
    valid_score = scorer.score(gts_valid, hypo_valid, valid_ids)  
    
  else: 
    valid_score = None
  
  if samples_test:

    gts_test, hypo_test, test_ids = make_template(samples_test, test)  
    print 'compute test set score:'
    test_score = scorer.score(gts_test, hypo_test, test_ids)

  else:
    test_score = None
  
  return valid_score, test_score

def compute_score(valid, test, word_idict, f_init, f_next, options, beam=5, save_result_dir = './exp/'):
  samples_valid, samples_test = generate_sample(valid, test, word_idict,
                                                f_init, f_next, options, beam=beam, 
                                                save_result_dir=save_result_dir)

  valid_score, test_score = score_with_cocoeval(samples_valid, samples_test, valid, test)
  scores_final = {}
  scores_final['valid'] = valid_score
  scores_final['test'] = test_score
  
  return scores_final
