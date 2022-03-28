import numpy as np
import matplotlib.pyplot as plt
from torch import gt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    max_x = min(prediction_box[2], gt_box[2])
    min_x = max(prediction_box[0], gt_box[0])
    max_y = min(prediction_box[3], gt_box[3])
    min_y = max(prediction_box[1], gt_box[1])
    x_inter = max_x - min_x
    y_inter = max_y - min_y
    if x_inter > 0 and y_inter > 0:
        areaOfIntersection = x_inter*y_inter
    else: 
        areaOfIntersection = 0

    # Compute union
    areaOfPred = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])
    areaOfGT = (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])
    iou = areaOfIntersection/(areaOfGT + areaOfPred - areaOfIntersection)
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) > 0:
        precision = num_tp/(num_tp + num_fp)
    else:
        precision = 1
    #raise NotImplementedError
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) > 0:
        recall = num_tp/(num_tp + num_fn)
    else:
        recall = 0
    #raise NotImplementedError
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """


    # Find all possible matches with a IoU >= iou threshold
    matches = []
    for i in range(len(prediction_boxes)):
        for j in range(len(gt_boxes)):
            iou = calculate_iou(prediction_boxes[i], gt_boxes[j])
            if iou>=iou_threshold:
                matches.append((i, j, iou))

    # Sort all matches on IoU in descending order
    if len(matches) == 0:
        return np.array([]), np.array([])


    sorted_matches = sorted(matches, key=lambda tup: tup[2], reverse = True)
    out_index = {}
    out_pred_index = []
    out_gt_index = []

    for index, tuple in enumerate(sorted_matches):
        if out_index.get(tuple[0]) is None:
            if (tuple[1] in out_index.values()) is False:
                out_index[tuple[0]] = tuple[1]
        
    for key in out_index:
        out_pred_index.append(key)
        out_gt_index.append(out_index[key])

    # Find all matches with the highest IoU threshold

    out_pred2 = np.zeros((len(out_pred_index), 4))
    out_gt2 = np.zeros((len(out_gt_index), 4))

    for i in range(len(out_pred_index)):
        out_pred2[i, :] = prediction_boxes[out_pred_index[i],:]
    for i in range(len(out_gt_index)):
        out_gt2[i, :] = gt_boxes[out_gt_index[i], :]

    return out_pred2, out_gt2


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # True positives are the number of boxes that was matching. 
    # True negative are the ## skal ikke ha ut denne for gir ikke mening.
    # False positiv are the predicted boxes that was not matched
    # False negative are the ground truth boxes that was not matched
    pred, gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    true_pos = len(pred)
    false_neg = len(gt_boxes) - true_pos
    false_pos = len(prediction_boxes) - true_pos

    d = {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}
    return d
    #raise NotImplementedError


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    numImages = len(all_prediction_boxes)

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(numImages):
        results = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        true_pos += results.get("true_pos")
        false_pos += results.get("false_pos")
        false_neg += results.get("false_neg")

    total_precision = calculate_precision(true_pos, false_pos, false_neg)
    total_recall = calculate_recall(true_pos, false_pos, false_neg)
    return (total_precision, total_recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []
    # YOUR CODE HERE
    for tresh_idx in range(len(confidence_thresholds)):
        conf_all_prediction_boxes = []
        

        for img in range(len(confidence_scores)):
            list = []
            for pred in range(len(all_prediction_boxes[img])):
                if confidence_scores[img][pred] >= confidence_thresholds[tresh_idx]:
                    list.append(all_prediction_boxes[img][pred])
                
            conf_all_prediction_boxes.append(np.array(list))
            precision, recall = calculate_precision_recall_all_images(conf_all_prediction_boxes, all_gt_boxes, iou_threshold)


        precisions.append(precision)
        recalls.append(recall)

        print(len(precisions))
        print(len(recalls))



    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    sum_precision = 0

    for level in recall_levels:
        precision = 0
        for n in range(len(recalls)):
            if (precisions[n] > precision) and (recalls[n] >= level):
                precision = precisions[n]

        sum_precision = sum_precision + precision

    AP = sum_precision / 11.0

    #average_precision = 0
    return AP


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
