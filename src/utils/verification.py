import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision import transforms


def calculate_accuracy(threshold, dist, actual_issame, useCos=False):
    '''
    if useCos = True, then view 'dist' variable as cos
    '''
    if useCos:
        predict_issame = np.greater(dist, threshold)
    else:
        predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc_attention(thresholds,
                            xCoses,
                            actual_issame, nrof_folds=10, pca=0):

    nrof_pairs = min(len(actual_issame), xCoses.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        cosines = xCoses

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            raise NotImplementedError

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold,
                cosines[train_set],
                actual_issame[train_set],
                useCos=True)
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   cosines[test_set],
                                   actual_issame[test_set],
                                   useCos=True)
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            cosines[test_set],
            actual_issame[test_set],
            useCos=True)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate_accuracy(xCoses, actual_issame, nrof_folds=10, pca=0):
    '''
    xCoses: np.array (# of pairs,)
    actual_issame: list (# of pairs,)
    '''
    # Calculate evaluation metrics
    thresholds = np.arange(-1.0, 1.0, 0.005)
    tpr, fpr, accuracy, best_thresholds = calculate_roc_attention(
        thresholds, xCoses,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
#                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
#     return tpr, fpr, accuracy, best_thresholds, val, val_std, far

    roc_curve_tensor = get_roc_curve(fpr, tpr)
    return accuracy, best_thresholds, roc_curve_tensor


def get_roc_curve(fpr, tpr):
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return roc_curve_tensor


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    # plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def getTFNPString(same, isSame_pred):
    title_str = 'LL'
    if same == 1 and int(isSame_pred) == 0:
        # not the same person but predicted the same]
        # False negative
        title_str = 'False_Negative'

    elif same == 0 and int(isSame_pred) == 1:
        # False positive
        title_str = 'False_Positive'

    elif same == 1 and int(isSame_pred) == 1:
        # True positive
        title_str = 'True_Positive'

    elif same == 0 and int(isSame_pred) == 0:
        # True negative
        title_str = 'True_Negative'
    return title_str


def checkTFPN(cos, is_same_label, threshold=0.2545):
    # 0.2545 is the threshold for xCosArcFace
    same = 1 if float(cos) > threshold else 0
    return getTFNPString(is_same_label, same)
