import numpy as np
from deepod.metrics import ts_metrics, point_adjustment, get_best_f1_and_threshold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


def create_range(start, end, step, decimal_places=10):
    """
    Create a list from start to end with specified step size
    
    Args:
        start (float): Starting value (inclusive)
        end (float): Ending value (inclusive)
        step (float): Step size
        decimal_places (int): Number of decimal places to round to
        
    Returns:
        list: List of numbers from start to end with step size
    """
    # Calculate number of steps (including endpoint)
    num_steps = int(round((end - start) / step)) + 1
    
    # Generate the list with proper rounding to avoid floating-point errors
    return [round(start + i * step, decimal_places) for i in range(num_steps)]

def certified_stats(y_true, scores, radiis, radii_thresholds, score_threshold, point_adj=False):
    c_scores = scores.copy()
    if point_adj:
        c_scores = point_adjustment(y_true, c_scores)
    y_pred = (c_scores >= score_threshold).astype(int)

    results = {
        'radii_stats': radii_stats(radiis),
        'overall': {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
        },
        'evasion': {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
        },
        'availability': {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
    }

    for radii_threshold in radii_thresholds:
        small_radii_mask = (radiis < radii_threshold)
        correct_pred_mask = (y_pred == y_true)
        positive_mask = (y_true == 1)
        negative_mask = (y_true == 0)

        # Overall stats
        overall_mask = small_radii_mask & correct_pred_mask
        overall_y_pred = y_pred.copy()
        overall_y_pred[overall_mask] = 1 - y_pred[overall_mask]
        results['overall']['accuracy'].append(accuracy_score(y_true, overall_y_pred))
        results['overall']['f1'].append(f1_score(y_true, overall_y_pred))
        results['overall']['precision'].append(precision_score(y_true, overall_y_pred))
        results['overall']['recall'].append(recall_score(y_true, overall_y_pred))

        # Evasion Attack: flip all the correctly predicted positive points with small radii   
        evasion_mask = small_radii_mask & correct_pred_mask & positive_mask
        evasion_y_pred = y_pred.copy()
        evasion_y_pred[evasion_mask] = 1 - y_pred[evasion_mask]
        results['evasion']['accuracy'].append(accuracy_score(y_true, evasion_y_pred))
        results['evasion']['f1'].append(f1_score(y_true, evasion_y_pred))
        results['evasion']['precision'].append(precision_score(y_true, evasion_y_pred))
        results['evasion']['recall'].append(recall_score(y_true, evasion_y_pred))
       
        # Avaliability Attack: flip all the correctly predicted negative points with small radii
        availability_mask = small_radii_mask & correct_pred_mask & negative_mask
        availability_y_pred = y_pred.copy()
        availability_y_pred[availability_mask] = 1 - y_pred[availability_mask]
        results['availability']['accuracy'].append(accuracy_score(y_true, availability_y_pred))
        results['availability']['f1'].append(f1_score(y_true, availability_y_pred))
        results['availability']['precision'].append(precision_score(y_true, availability_y_pred))
        results['availability']['recall'].append(recall_score(y_true, availability_y_pred))
       
    return results


def certified_f1_p_r(y_true, scores, radiis, radii_thresholds, score_threshold, point_adj=False):
    f1s = []
    ps = []
    rs = []
    for radii_threshold in radii_thresholds:
        c_scores = scores.copy()
        if point_adj:
            c_scores = point_adjustment(y_true, c_scores)
        y_pred = (c_scores >= score_threshold).astype(int)
        
        # Flip ONLY CORRECTLY predicted labels for points with small radiis
        small_radii_mask = (radiis < radii_threshold)
        correct_pred_mask = (y_pred == y_true)
        
        # Combined mask for points that are both correctly predicted AND have small radius
        flip_mask = small_radii_mask & correct_pred_mask
        
        # Flip only those points
        y_pred[flip_mask] = 1 - y_pred[flip_mask]

        f1s.append(f1_score(y_true, y_pred))
        ps.append(precision_score(y_true, y_pred))
        rs.append(recall_score(y_true, y_pred))

    return f1s, ps, rs
        
def radii_stats(radiis):
    stats = {}
    stats['mean'] = np.mean(radiis)
    stats['std'] = np.std(radiis)
    stats['min'] = np.min(radiis)
    stats['max'] = np.max(radiis)
    stats['proportion'] = np.sum(radiis > 0.0) / len(radiis)
    return stats



