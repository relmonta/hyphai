import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, cmatrices: np.ndarray = None) -> np.ndarray:
    """
    Compute the accuracy between two arrays for each time step and each label.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        cmatrices (np.ndarray, optional): Confusion matrices, array of shape (T, C, C). Defaults to None.

    Returns:
        np.ndarray: Accuracy for each time step and each label, array of shape (T, C),
        where T is the number of time steps and C is the number of classes.

        Accuracy_j = (TP_j + TN_j) / (TP_j + TN_j + FP_j + FN_j) for all j in {1, 2, ..., C}.
    """
    T, X, Y = y_true.shape
    C = y_true.max() + 1  # The number of classes is derived from y_true
    list_acc = np.zeros((T, C))

    for t in range(T):
        # Compute the confusion matrix
        if cmatrices is not None:
            conf_matrix = cmatrices[t]
        else:
            conf_matrix = confusion_matrix(y_true[t].flatten(), y_pred[t].flatten(), labels=list(range(C))).astype(np.float64)

        # Compute the accuracy for each class
        for j in range(C):
            # Indices of the other classes
            indices = np.delete(np.arange(C), j)
            false = np.sum(conf_matrix[indices, j]) + np.sum(conf_matrix[j, indices])
            correct = np.sum(conf_matrix) - false
            list_acc[t, j] = np.divide(correct, (correct + false), out=np.zeros_like(correct), where=(correct + false) != 0)

    return list_acc

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-2) -> np.ndarray:
    r"""
    Compute the Intersection over Union (IoU) score between two arrays for each time step and each label.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        epsilon (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-2.

    Returns:
        np.ndarray: IoU score for each time step and each label, array of shape (T, C),
        where T is the number of time steps and C is the number of classes.

    IoU is defined as: :math:`IoU_c = \frac{|Y_c \cap Y_{\text{pred}_c}| + \epsilon}{|Y_c \cup Y_{\text{pred}_c}| + \epsilon}`,
    where Y_c represents the ground truth mask for class c and Y_pred_c represents the predicted mask for class c.
    """
    iou_list = np.zeros((y_true.shape[0], y_true.max() + 1))  # Number of classes is determined by the maximum value in y_true
    for t in range(y_true.shape[0]):
        for c in range(y_true.max() + 1):
            y_true_c = 1 * (y_true[t] == c)
            y_pred_c = 1 * (y_pred[t] == c)
            intersection = np.sum(y_true_c * y_pred_c)
            union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
            iou_list[t, c] = (intersection + epsilon) / (union + epsilon)
    return iou_list

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, cmatrices: np.ndarray = None) -> tuple:
    """
    Compute the F1 score, precision, and recall between two arrays for each time step and each label.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        cmatrices (np.ndarray, optional): Confusion matrices, array of shape (T, C, C).
            Defaults to None.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): F1 score, precision, and recall for each time step and each label,
        arrays of shape (T, C), where T is the number of time steps and C is the number of classes.

    F1 score is the harmonic mean of precision and recall, defined as:
    F1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c),
    where precision_c and recall_c are the precision and recall for class c, respectively.
    """
    n_classes = y_true.max() + 1  # Number of classes is determined by the maximum value in y_true
    f1 = np.zeros((y_true.shape[0], n_classes))
    precisions = np.zeros((y_true.shape[0], n_classes))
    recalls = np.zeros((y_true.shape[0], n_classes))
    for i in range(y_true.shape[0]):
        # Compute the confusion matrix
        if cmatrices is not None:
            conf_matrix = cmatrices[i]
        else:
            conf_matrix = confusion_matrix(y_true[i].flatten(), y_pred[i].flatten(), labels=list(range(n_classes))).astype(np.float64)
        for c in range(n_classes):
            tp = conf_matrix[c, c]
            fp = np.sum(conf_matrix[c, :]) - tp
            fn = np.sum(conf_matrix[:, c]) - tp
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            f1[i, c] = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
            precisions[i, c] = precision
            recalls[i, c] = recall
    return f1, precisions, recalls

def hausdorff(y_true: np.ndarray, y_pred: np.ndarray, radius: float) -> np.ndarray:
    r"""
    Compute the Hausdorff distance between two arrays for each time step and each label.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        radius (float): Maximum distance between two points.

    Returns:
        np.ndarray: Average Hausdorff distance for each time step and each label,
        array of shape (T, C), where T is the number of time steps and C is the number of classes.

    The Hausdorff distance measures the similarity between two sets of points (X and Y) and is defined as:
    :math:`AHD(X, Y)=\max\left(\frac{1}{|X|} \sum_{p \in X} \min_{q \in Y} \|p - q\|, \frac{1}{|Y|} \sum_{q \in Y} \min_{p \in X} \|p - q\|\right)`,
    where :math:`|X|` and :math:`|Y|` are the number of points in sets X and Y, respectively.
    """
    n_classes = y_true.max() + 1  # Number of classes is determined by the maximum value in y_true
    hausdorff_list = np.zeros((y_true.shape[0], n_classes))
    for c in range(n_classes):
        for t in range(y_true.shape[0]):
            # Get the border points of the ground truth and prediction for class c
            y_true_c = (y_true[t] == c) * 1
            y_pred_c = (y_pred[t] == c) * 1
            tmp = np.where(y_true_c == 1)
            y_true_c_points = np.array([tmp[0], tmp[1]]).T
            tmp = np.where(y_pred_c == 1)
            y_pred_c_points = np.array([tmp[0], tmp[1]]).T
            if y_pred_c_points.size == 0 or y_true_c_points.size == 0:
                hausdorff_list[t, c] = radius
            else:
                # Compute the Hausdorff distance
                hausdorff_list[t, c] = average_hausdorff_distance(y_true_c_points, y_pred_c_points, radius)
    return hausdorff_list

def avg_accuracy(y_true: np.ndarray, y_pred: np.ndarray, cmatrices: np.ndarray = None) -> np.ndarray:
    """
    Compute the average accuracy between two arrays for each time step.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        cmatrices (np.ndarray, optional): Confusion matrices, array of shape (T, C, C).
            Defaults to None.

    Returns:
        np.ndarray: Accuracy for each time step, array of shape (T,).
    """
    n_classes = y_true.max() + 1  # Number of classes is determined by the maximum value in y_true
    list_acc = np.zeros((y_true.shape[0]))
    for i in range(y_true.shape[0]):
        # Compute the confusion matrix
        if cmatrices is not None:
            conf_matrix = cmatrices[i]
        else:
            conf_matrix = confusion_matrix(y_true[i].flatten(), y_pred[i].flatten(), labels=list(range(n_classes))).astype(np.float64)
        # Compute the accuracy
        list_acc[i] = np.diag(conf_matrix).sum() / conf_matrix.sum()
    return list_acc

def avg_iou_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'micro', epsilon: float = 1e-2) -> np.ndarray:
    """
    Compute the average Intersection over Union (IoU / CSI / Jaccard index) score between two arrays for each time step.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        average (str, optional): Averaging method, either 'macro' or 'micro'. Defaults to 'micro'.
        epsilon (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-2.

    Returns:
        np.ndarray: IoU score for each time step, array of shape (T,).

    The 'average' parameter specifies the averaging method for multiple classes:
    - 'macro': Computes the IoU for each class and averages them.
    - 'micro': Computes the IoU by considering all classes together.
    """
    n_classes = y_true.max() + 1  # Number of classes is determined by the maximum value in y_true
    iou_list = np.zeros((y_true.shape[0]))
    if average == 'macro':
        for t in range(y_true.shape[0]):
            tmp_ious = np.zeros((n_classes))
            for c in range(n_classes):
                y_true_c = 1 * (y_true[t] == c)
                y_pred_c = 1 * (y_pred[t] == c)
                intersection = np.sum(y_true_c * y_pred_c)
                union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
                tmp_ious[c] = (intersection + epsilon) / (union + epsilon)
            iou_list[t] = np.mean(tmp_ious)
    else:
        for t in range(y_true.shape[0]):
            intersections = np.zeros((n_classes))
            unions = np.zeros((n_classes))
            for c in range(n_classes):
                y_true_c = 1 * (y_true[t] == c)
                y_pred_c = 1 * (y_pred[t] == c)
                intersections[c] = np.sum(y_true_c * y_pred_c)
                unions[c] = np.sum(y_true_c) + np.sum(y_pred_c) - intersections[c]
            iou_list[t] = np.sum(intersections) / (np.sum(unions) + epsilon)
    return iou_list

def avg_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'micro', cmatrices: np.ndarray = None) -> tuple:
    """
    Compute the averaged F1 score, precision, and recall for each time step in the given data.

    Args:
        y_true (np.ndarray): Ground truth, array of shape (T, X, Y).
        y_pred (np.ndarray): Prediction, array of shape (T, X, Y).
        average (str, optional): Averaging method, either 'macro' or 'micro'. Defaults to 'micro'.
        cmatrices (np.ndarray, optional): Confusion matrices, array of shape (T, C, C).
            Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following arrays:
            - Averaged F1 score for each time step, array of shape (T,).
            - Averaged precision for each time step, array of shape (T,).
            - Averaged recall for each time step, array of shape (T,).

    The 'average' parameter specifies the averaging method for multiple classes:
    - 'macro': Computes the F1 score, precision, and recall for each class and averages them.
    - 'micro': Computes the F1 score, precision, and recall by considering all classes together.
    """
    n_classes = y_true.max() + 1  # Number of classes is determined by the maximum value in y_true
    list_f1 = np.zeros((y_true.shape[0]))
    list_precisions = np.zeros((y_true.shape[0]))
    list_recalls = np.zeros((y_true.shape[0]))
    for i in range(y_true.shape[0]):
        # Compute the confusion matrix
        if cmatrices is not None:
            conf_matrix = cmatrices[i]
        else:
            conf_matrix = confusion_matrix(y_true[i].flatten(), y_pred[i].flatten(), labels=list(range(n_classes))).astype(np.float64)
        # Compute the F1 score
        if average == "macro":
            # Compute precision and recall
            precisions = np.zeros(n_classes)
            recalls = np.zeros(n_classes)
            for c in range(n_classes):
                tp = conf_matrix[c, c]
                fp = np.sum(conf_matrix[c, :]) - tp
                fn = np.sum(conf_matrix[:, c]) - tp
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                precisions[c] = precision
                recalls[c] = recall
            ma_precision = np.mean(precisions)
            ma_recall = np.mean(recalls)
            list_f1[i] = 2 * ma_precision * ma_recall / (ma_precision + ma_recall) if (ma_precision + ma_recall) != 0 else 0.0
            list_precisions[i] = ma_precision
            list_recalls[i] = ma_recall
        else:
            # F1 = sum(cm_ii) / sum(cm_ij)
            tp_sum = np.sum(np.diag(conf_matrix))
            cm_sum = np.sum(conf_matrix)
            f1_micro = tp_sum / cm_sum if cm_sum != 0 else 0.0
            list_f1[i] = f1_micro
            list_precisions[i] = f1_micro
            list_recalls[i] = f1_micro

    return list_f1, list_precisions, list_recalls

def average_hausdorff_distance(points1: np.ndarray, points2: np.ndarray, max_d: float) -> float:
    r"""
    Compute the Average Hausdorff distance between two sets of points (X and Y).

    Args:
        points1 (np.ndarray): Array of shape (:math:`|X|`, 2) containing points of set X.
        points2 (np.ndarray): Array of shape (:math:`|Y|`, 2) containing points of set Y.
        max_d (float): Maximum distance between two points to be considered.

    Returns:
        float: The Average Hausdorff distance between points1 and points2.

    The Average Hausdorff distance is the Hausdorff distance averaged over all points.
    Hausdorff distance measures the similarity between two sets of points (X and Y) and is defined as:
    :math:`AHD(X, Y)=\max\left(\frac{1}{|X|} \sum_{p \in X} \min_{q \in Y} \|p - q\|, \frac{1}{|Y|} \sum_{q \in Y} \min_{p \in X} \|p - q\|\right)`,
    where :math:`|X|` and :math:`|Y|` are the number of points in sets X and Y, respectively.
    """
    # Compute the pairwise distances between all points in both arrays
    distances = cdist(points1, points2)

    # Compute the directed Hausdorff distance from points1 to points2
    average_hd_dist_1 = np.min(distances, axis=1)
    average_hd_dist_1[average_hd_dist_1 > max_d] = max_d
    average_hd_dist_1 = average_hd_dist_1.mean()

    # Compute the directed Hausdorff distance from points2 to points1
    average_hd_dist_2 = np.min(distances, axis=0)
    average_hd_dist_2[average_hd_dist_2 > max_d] = max_d
    average_hd_dist_2 = average_hd_dist_2.mean()

    # Return the maximum of the two directed Hausdorff distances
    average_hd_dist = max(average_hd_dist_1, average_hd_dist_2)

    return average_hd_dist
