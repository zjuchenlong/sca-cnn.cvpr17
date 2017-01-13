This directory **exp** is used for storing experiments results

In each experiment directory (the default is test_1), it contains following files:
- **model.npz.pkl**: training configuration parameters for this experiment.
- **history_errs.txt**: train errors in each iteration.
- **history_score.txt**: store the evaluation scores of validation set and test set.
- **model_best_bleu.npz**: store training parameters with best evaluation scores.
- **model_best_so_far.npz**: store training parameters with minimal loss.
- **valid_samples.txt**: generated captions on validation set for the recently evaluation.
- **test_samples.txt**: generated captions on test set for the recently evaluation.
