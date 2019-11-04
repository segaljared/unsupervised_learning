import sklearn.model_selection as sklearn_m
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn.datasets
import csv
import numpy
import cluster_analysis
import dimensionality
import neural_net
import complexity_analysis
import os

def main():
    if not os.path.exists('figures'):
        os.makedirs('figures')
    if not os.path.exists('data'):
        os.makedirs('data')

    features, classes = load_digits_data()
    run_tests('Digits', features, classes, True, (10,), max_k=15, run_nn_analysis=False)

    features, classes = load_default_data()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    run_tests('Credit Card Default', scaled_features, classes, True, (15,5))

def run_tests(name, features, classes, run_nn, hidden_units, max_k=10, run_nn_analysis=False, k_means_k=None, em_k=None, pca_k=None, ica_k=None, rca_k=None, other_k=None):
    test_size = int(len(features) * 0.2)
    training_features, test_features, training_classes, test_classes = sklearn_m.train_test_split(features, classes, test_size=test_size, train_size=(len(features) - test_size), random_state=50207)
    
    if run_nn_analysis:
        print('Starting base analysis of the NN...')
        hidden_units = neural_net.perform_complexity_analysis(training_features,
                                                            training_classes,
                                                            '{} NN Base'.format(name),
                                                            [(5,), (10,), (15,), (30,), (50,), (5,5), (10,5), (15,5), (15,10), 
                                                                (30,5), (30,10), (30,15), (15,10,5), (30,15,5)])

        print('Best hidden units: {}'.format(hidden_units))

    neural_net.run_neural_net_analysis('{} NN Base'.format(name), training_features, training_classes, test_features, test_classes, hidden_units)

    # part 1
    print('Performing KMeans cluster analysis', end='', flush=True)
    k, kmeans_clusterers = cluster_analysis.run_k_cluster_analysis_k_means(name,
                                                                           training_features,
                                                                           training_classes,
                                                                           max_k=max_k)
    if k_means_k is None:
        k_means_k = k

    print('Performing EM cluster analysis', end='', flush=True)
    k, em_clusterers = cluster_analysis.run_k_cluster_analysis_EM(name,
                                                                  training_features,
                                                                  training_classes,
                                                                  max_k=max_k)
    if em_k is None:
        em_k = k

    # part 2
    dim_reduction = {}

    print('Performing analysis of Decision Tree', end='', flush=True)
    tree = dimensionality.run_decision_tree_dimensionality_reduction('{} DT'.format(name),
                                                                     training_features,
                                                                     training_classes,
                                                                     chosen_k=other_k)
    dim_reduction['DT'] = tree

    print('Performing analysis of PCA', end='', flush=True)
    pca = dimensionality.run_dimensionality_reduction_pca('{} PCA'.format(name),
                                                          training_features,
                                                          training_classes,
                                                          chosen_k=pca_k)
    dim_reduction['PCA'] = pca

    print('Performing analysis of ICA', end='', flush=True)
    ica = dimensionality.run_dimensionality_reduction_ica('{} ICA'.format(name),
                                                          training_features,
                                                          training_classes,
                                                          chosen_k=ica_k)
    dim_reduction['ICA'] = ica

    print('Performing analysis of RCA', end='', flush=True)
    rca = dimensionality.run_dimensionality_reduction_rca('{} RCA'.format(name),
                                                          training_features,
                                                          training_classes,
                                                          chosen_k=rca_k)
    dim_reduction['RCA'] = rca

    #part 3
    total_difference_by_dr_kmeans = {}
    total_difference_by_dr_em = {}
    for dim_reducer in dim_reduction:
        reducer = dim_reduction[dim_reducer]
        full_name = '{} ({})'.format(name, dim_reducer)
        reduced_features = reducer.transform(training_features)
        print('Running KMeans cluster analysis after feature reduction ({})'.format(dim_reducer), end='', flush=True)
        _, clusterers = cluster_analysis.run_k_cluster_analysis_k_means('{} KMeans'.format(full_name),
                                                                        reduced_features,
                                                                        training_classes,
                                                                        max_k=max_k)
        total_difference = cluster_analysis.compare_clusterers(full_name, kmeans_clusterers, clusterers, training_features, reduced_features, training_classes, 2)
        total_difference_by_dr_kmeans[dim_reducer] = total_difference

        print('Running EM cluster analysis after feature reduction ({})'.format(dim_reducer), end='', flush=True)
        _, clusterers = cluster_analysis.run_k_cluster_analysis_EM('{} EM'.format(full_name),
                                                                   reduced_features,
                                                                   training_classes,
                                                                   max_k=max_k)
        total_difference = cluster_analysis.compare_clusterers(full_name, em_clusterers, clusterers, training_features, reduced_features, training_classes, 2)
        total_difference_by_dr_em[dim_reducer] = total_difference

    cluster_analysis.plot_total_difference('{} KMeans'.format(name), total_difference_by_dr_kmeans)
    cluster_analysis.plot_total_difference('{} EM'.format(name), total_difference_by_dr_em)

    if run_nn:
        # part 4
        for dim_reducer in dim_reduction:
            reducer = dim_reduction[dim_reducer]
            full_name = '{} ({})'.format(name, dim_reducer)
            reduced_features = reducer.transform(training_features)
            reduced_test_features = reducer.transform(test_features)

            print('Performing NN analysis on data reduced with {}...'.format(dim_reducer))
            neural_net.run_neural_net_analysis(full_name,
                                               reduced_features,
                                               training_classes,
                                               reduced_test_features,
                                               test_classes,
                                               hidden_units)
        
        # part 5
        print('Performing NN analysis using KMeans cluster...')
        full_name = '{} with K Means'.format(name)
        clusterer = cluster_analysis.run_k_cluster_k_means(reduced_features, training_classes, k_means_k)
        clusters = clusterer.predict(reduced_features)
        reduced_features_w_cluster = numpy.append(reduced_features, clusters.reshape(len(clusters), 1), axis=1)
        clusters = clusterer.predict(reduced_test_features)
        reduced_test_features_w_cluster = numpy.append(reduced_test_features, clusters.reshape(len(clusters), 1), axis=1)
        neural_net.run_neural_net_analysis(full_name,
                                            reduced_features_w_cluster,
                                            training_classes,
                                            reduced_test_features_w_cluster,
                                            test_classes,
                                            hidden_units)
        
        print('Performing NN analysis using EM cluster...')
        full_name = '{} with EM'.format(name)
        clusterer = cluster_analysis.run_k_cluster_EM(reduced_features, training_classes, em_k)
        clusters = clusterer.predict(reduced_features)
        reduced_features_w_cluster = numpy.append(reduced_features, clusters.reshape(len(clusters), 1), axis=1)
        clusters = clusterer.predict(reduced_test_features)
        reduced_test_features_w_cluster = numpy.append(reduced_test_features, clusters.reshape(len(clusters), 1), axis=1)
        neural_net.run_neural_net_analysis(full_name,
                                            reduced_features_w_cluster,
                                            training_classes,
                                            reduced_test_features_w_cluster,
                                            test_classes,
                                            hidden_units)


def load_default_data():
    features = []
    classes = []
    with open('default_credit_card.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for csv_row in reader:
            class_index = len(csv_row) - 1
            row = csv_row[0:class_index]
            features.append(row)
            classes.append(csv_row[class_index])
    return numpy.array(features).astype(float), numpy.array(classes).astype(float)


def load_digits_data():
    digits = sklearn.datasets.load_digits()
    features = digits.images.reshape((len(digits.images), -1))
    classes = digits.target
    return features, classes

if __name__ == "__main__":
    main()
