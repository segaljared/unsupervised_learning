from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from decision_tree_reducer import get_k_depth_values, DecisionTreeDimReducer
import scipy.stats
import numpy
import matplotlib.pyplot as plot
from sklearn.utils.extmath import safe_sparse_dot
import project_constants
from yellowbrick.utils import KneeLocator

COLORS = ['firebrick', 'orangered', 'gold', 'yellowgreen', 'darkslategrey', 'deepskyblue', 'cornflowerblue', 'navy', 'darkorchid', 'crimson']

def run_dimensionality_reduction_pca(name, features, classes, min_k=2, max_k=None, chosen_k=None, random_state=6126540):
    def create_pca(k, r_state):
        return PCA(n_components=k, random_state=r_state)

    def apply_metric(transformer, features):
        return numpy.sum(transformer.explained_variance_ratio_)

    def choose_from_metric(metric, k_values):
        if chosen_k is not None:
            return chosen_k
        knee_locator = KneeLocator(k_values, metric)
        if knee_locator.knee is None:
            return k_values[numpy.argmax(metric)]
        else:
            return knee_locator.knee
    
    return run_dimensionality_reduction(name,
                                        create_pca,
                                        'explained variance',
                                        apply_metric,
                                        choose_from_metric,
                                        features,
                                        classes,
                                        min_k=min_k,
                                        max_k=max_k,
                                        perform_reconstruction_error=True,
                                        random_state=random_state)

def run_dimensionality_reduction_ica(name, features, classes, min_k=2, max_k=None, chosen_k=None, random_state=6126540):
    def create_ica(k, r_state):
        return FastICA(n_components=k, random_state=r_state, max_iter=5000)

    def apply_metric(transformer, features):
        reduced_features = transformer.transform(features)
        kurtosis = scipy.stats.kurtosis(reduced_features)
        return numpy.mean(kurtosis)

    def choose_from_metric(metric, k_values):
        if chosen_k is not None:
            return chosen_k
        return k_values[numpy.argmax(metric)]

    ica = run_dimensionality_reduction(name,
                                       create_ica,
                                       'average kurtosis',
                                       apply_metric,
                                       choose_from_metric,
                                       features,
                                       classes,
                                       min_k=min_k,
                                       max_k=max_k,
                                       random_state=random_state)

    if chosen_k is not None:
        transformed_features = ica.transform(features)
        plot_ica_histogram(name, transformed_features, classes)

    return ica
            

def run_dimensionality_reduction_rca(name, features, classes, min_k=2, max_k=None, chosen_k=None, random_state=6126540):
    def create_rca(k, r_state):
        return SparseRandomProjection(n_components=k, random_state=r_state)

    def apply_metric(transformer, features):
        transformed_features = transformer.transform(features)
        reconstructed_features = safe_sparse_dot(transformed_features, transformer.components_, dense_output=transformer.dense_output)
        return numpy.sum((numpy.array(features) - reconstructed_features)**2)

    def choose_from_metric(metric, k_values):
        if chosen_k is not None:
            return chosen_k
        knee_locator = KneeLocator(k_values, metric, curve_nature='convex', curve_direction='decreasing')
        if knee_locator.knee is None:
            return k_values[numpy.argmin(metric)]
        else:
            return knee_locator.knee

    return run_dimensionality_reduction(name,
                                        create_rca,
                                        'reconstruction error',
                                        apply_metric,
                                        choose_from_metric,
                                        features,
                                        classes,
                                        min_k=min_k,
                                        max_k=max_k,
                                        average_over=20,
                                        random_state=random_state)

def run_dimensionality_reduction_lda(name, features, classes, min_k=2, max_k=None, chosen_k=None, random_state=6126540):
    def create_lda(k, r_state):
        return SparseRandomProjection(n_components=k, random_state=r_state)

    def apply_metric(transformer, features):
        transformed_features = transformer.transform(features)
        reconstructed_features = safe_sparse_dot(transformed_features, transformer.components_, dense_output=transformer.dense_output)
        return numpy.sum((numpy.array(features) - reconstructed_features)**2)

    def choose_from_metric(metric, k_values):
        if chosen_k is not None:
            return chosen_k
        return numpy.argmin(metric)

    return run_dimensionality_reduction(name, create_lda, 'reconstruction error', apply_metric, choose_from_metric, features, classes, min_k=min_k, max_k=max_k, random_state=random_state)

def run_dimensionality_reduction(name, create_transformer, metric_name, apply_metric, choose_from_metric, features, classes, min_k=2, max_k=None, average_over=1, perform_reconstruction_error=False, random_state=6126540):
    if max_k is None:
        max_k = features.shape[1] - 1
    
    metric = []
    metric_variance = []
    k_values = []
    reconstruction_errors = []
    for k in range(min_k, max_k + 1):
        k_values.append(k)
        print('.', end='', flush=True)
        metric_for_k = []
        recon_error_for_k = []
        numpy.random.seed(random_state)
        for _ in range(0, average_over):
            transformer = create_transformer(k, numpy.random.randint(100000000))

            transformer.fit(features)

            metric_for_k.append(apply_metric(transformer, features))
            if perform_reconstruction_error:
                recon_error_for_k.append(pca_reconstruction_error(transformer, features))
        metric.append(numpy.mean(metric_for_k))
        if perform_reconstruction_error:
            reconstruction_errors.append(numpy.mean(recon_error_for_k))
        if average_over > 1:
            metric_variance.append(numpy.std(metric_for_k))
    print('.')

    best_k = choose_from_metric(metric, k_values)
    if average_over > 1:
        plot_metric(name, metric_name, metric, k_values, best_k, metric_variance=metric_variance)
    else:
        plot_metric(name, metric_name, metric, k_values, best_k)

    if perform_reconstruction_error:
        plot_metric(name, 'reconstruction error', reconstruction_errors, k_values, best_k)

    best_transformer = create_transformer(best_k, random_state)
    best_transformer.fit(features)
    return best_transformer

def run_decision_tree_dimensionality_reduction(name, features, classes, min_k=2, max_k=None, min_depth=2, max_depth=None, chosen_k=None, random_state=6126540):
    if max_k is None:
        max_k = features.shape[1] - 1
    if max_depth is None:
        max_depth = features.shape[1]
    
    scores = []
    k_values = []
    k_value_strings = []
    k_to_depth = {}
    for k, depth, score in get_k_depth_values(features, classes, min_depth, max_depth, random_state):
        k_values.append(k)
        scores.append(score)
        k_to_depth[k] = depth
        k_value_strings.append('{}({})'.format(k, depth))
    print('.')
    
    if chosen_k is not None:
        best_k = chosen_k
    else:
        knee_locator = KneeLocator(k_values, scores)
        if knee_locator.knee is None:
            best_k = k_values[numpy.argmax(scores)]
        else:
            best_k = knee_locator.knee

    plot_metric(name, 'f1 score', scores, k_values, best_k)

    best_depth = k_to_depth[best_k]
    transformer = DecisionTreeDimReducer(best_depth, random_state)
    transformer.fit(features, classes)
    return transformer

def plot_metric(name, metric_name, metric, k_values, best_k, metric_variance=None):
    plot.clf()
    plot.title('{} for {}'.format(metric_name, name))
    plot.ylabel('{}'.format(metric_name))
    plot.xlabel('# of features')
    if metric_variance is not None:
        plot.errorbar(k_values, metric, yerr=metric_variance, marker=".")
    else:
        plot.plot(k_values, metric, marker=".")
    plot.axvline(best_k, ls='--', color='darkorange', alpha=0.7, lw=1)
    k_index = k_values.index(best_k)
    min_metric = numpy.min(metric)
    best_k_metric = metric[k_index]
    if abs(numpy.max(metric) - best_k_metric) > abs(min_metric - best_k_metric):
        pos_y = best_k_metric + abs(best_k_metric - min_metric) * 0.1
    else:
        pos_y = best_k_metric - abs(best_k_metric - min_metric) * 0.1
    plot.text(best_k, pos_y, ' k = {}'.format(best_k))
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_{}_metric'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower(), metric_name.replace(' ', '_')))

def plot_ica_histogram(name, transformed_features, classes):
    for f in range(0, transformed_features.shape[1]):
        feature = transformed_features[:,f]
        by_class = []
        for c in numpy.unique(classes):
            indices = numpy.nonzero(classes == c)
            by_class.append(feature[indices])
        plot.clf()
        plot.hist(numpy.array(by_class), bins=200, stacked=True)
        plot.pause(project_constants.PAUSE)
        plot.savefig('figures/{}_ica_by_axis_{}_histogram'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower(), f))

def pca_reconstruction_error(transformer, features):
    transformed_features = transformer.transform(features)
    reconstructed_features = transformer.inverse_transform(transformed_features)
    return numpy.sum((numpy.array(features) - reconstructed_features)**2)
