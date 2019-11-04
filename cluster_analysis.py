from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, homogeneity_completeness_v_measure
import numpy
import matplotlib.pyplot as plot
import project_constants


COLORS = ['firebrick', 'orangered', 'gold', 'yellowgreen', 'darkslategrey', 'deepskyblue', 'cornflowerblue', 'navy', 'darkorchid', 'crimson', 'darkred', 'chocolate', 'cyan', 'gray', 'violet']

def run_k_cluster_analysis_k_means(name, features, classes, max_k=10, random_state=6126540):
    def create_k_means(k):
        return KMeans(n_clusters=k, random_state=random_state)

    return run_k_cluster_analysis('{} KMeans'.format(name), features, classes, create_k_means, max_k=max_k, random_state=random_state)

def run_k_cluster_analysis_EM(name, features, classes, max_k=10, random_state=6126540):
    def create_gaussian_mixture(k):
        return GaussianMixture(n_components=k, random_state=random_state)

    return run_k_cluster_analysis('{} EM'.format(name), features, classes, create_gaussian_mixture, max_k=max_k, random_state=random_state)

def run_k_cluster_k_means(features, classes, k, random_state=6126540):
    k_means =  KMeans(n_clusters=k, random_state=random_state)
    k_means.fit(features)
    return k_means

def run_k_cluster_EM(features, classes, k, random_state=6126540):
    em = GaussianMixture(n_components=k, random_state=random_state)
    em.fit(features)
    return em

def run_k_cluster_analysis(name, features, classes, create_learner, max_k=10, random_state=6126540):
    scores = []
    average_silhouette_values = []
    by_cluster_silhouettes = []
    k_values = []
    by_cluster_class_values = []
    homogeneity_by_cluster = []
    completeness_by_cluster = []
    clusterers = []
    for k in range(2, max_k + 1):
        print('.', end='', flush=True)
        k_values.append(k)

        learner = create_learner(k)
        learner.fit(features)
        clusterers.append(learner)

        labels = learner.predict(features)
        score = learner.score(features)
        scores.append(score)

        homogeneity, completeness, _ = homogeneity_completeness_v_measure(classes, labels)
        homogeneity_by_cluster.append(homogeneity)
        completeness_by_cluster.append(completeness)

        by_cluster_class_values.append(compute_class_values(features, labels, classes, k))

        average_silhouette, by_cluster_silhouette = compute_silhouette_scores(features, labels, k)
        average_silhouette_values.append(average_silhouette)
        by_cluster_silhouettes.append(by_cluster_silhouette)
    print('.')
    plot_elbow_graph(name, scores, k_values)
    plot_homogeneity_and_completeness(name, homogeneity_by_cluster, completeness_by_cluster, k_values)
    plot_by_cluster_class_values(name, by_cluster_class_values, classes)
    best_k = k_values[numpy.argmax(average_silhouette_values)]
    plot_silhouette(name, average_silhouette_values, by_cluster_silhouettes, k_values, best_k)

    plot.clf()

    return best_k, clusterers

def compute_class_values(features, labels, classes, k):
    by_cluster_class_values = []
    unique_classes = numpy.unique(classes)
    for i in range(0, k):
        cluster_i_indices = numpy.argwhere(labels == i)
        class_values = classes[cluster_i_indices]
        class_counts = []
        for class_value in unique_classes:
            count = numpy.sum(class_values == class_value)
            class_counts.append(count)
        by_cluster_class_values.append(class_counts)
    return by_cluster_class_values

def plot_by_cluster_class_values(name, by_cluster_class_values, classes):
    # based this off of https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    unique_classes = numpy.unique(classes)
    for by_cluster_class_value in by_cluster_class_values:
        x = numpy.arange(len(by_cluster_class_value))
        width = 1 / (len(unique_classes) + 1)
        plot.clf()
        plot.title('Classes in each cluster for {}'.format(name))
        plot.ylabel('# of class instances')
        plot.xticks(x)
        plot.xlabel('cluster')
        cluster_values = numpy.array(by_cluster_class_value)
        for i, class_value in enumerate(unique_classes):
            offset = 0.5 - (i + 1) * width
            plot.bar(x - offset, cluster_values[:, i], width=width, label=class_value, color=COLORS[i % len(COLORS)])
        for i in range(0, len(x) - 1):
            plot.axvline((x[i] + x[i + 1]) / 2, ls='--', color='blue', alpha=0.7, lw=1)
        legend = plot.legend(loc='upper left', bbox_to_anchor=(1,1))
        plot.pause(project_constants.PAUSE)
        plot.savefig('figures/{}_by_cluster_label_{}_clusters'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower(), len(by_cluster_class_value)), bbox_extra_artists=(legend,), bbox_inches='tight')

def compute_silhouette_scores(features, labels, k):
    average_silhouette = silhouette_score(features, labels)

    by_sample_silhouette = silhouette_samples(features, labels)

    by_cluster_silhouette = []
    for i in range(0, k):
        cluster_i_indices = numpy.argwhere(labels == i)
        cluster_silhouette_values = by_sample_silhouette[cluster_i_indices]
        cluster_silhouette_values = cluster_silhouette_values.reshape((cluster_silhouette_values.shape[0],))
        cluster_silhouette_values.sort()
        by_cluster_silhouette.append(cluster_silhouette_values)
    return average_silhouette, by_cluster_silhouette

def plot_silhouette(name, average_silhouette_values, by_cluster_silhouettes, k_values, best_k):
    plot.clf()
    plot.title('Average silhouette value for {}'.format(name))
    plot.ylabel('silhouette value')
    plot.xlabel('# of clusters')
    plot.plot(k_values, average_silhouette_values, marker=".")
    plot.axvline(best_k, ls='--', color='darkorange', alpha=0.7, lw=1)
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_average_silhouette'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))

    # based plotting technique on this: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    for i, k in enumerate(k_values):
        by_cluster_silhouette = by_cluster_silhouettes[i]

        y_separation = round(numpy.array(by_cluster_silhouette).size * 0.01)
        y_start = y_separation
        plot.clf()
        plot.title('Silhouette plot for {} with {} clusters'.format(name, k))
        plot.ylabel('cluster label')
        plot.xlabel('silhouette value')
        for label, cluster_silhouette in enumerate(by_cluster_silhouette):
            size = len(cluster_silhouette)
            y_end = y_start + size
            color = COLORS[label % len(COLORS)]
            plot.fill_betweenx(numpy.arange(y_start, y_end), 0, cluster_silhouette, color=color, alpha=0.75)
            plot.text(-0.05, (y_start + y_end) / 2, str(label))
            y_start = y_end + y_separation
        plot.axvline(average_silhouette_values[i], ls='--', color='black', alpha=0.7, lw=1)
        plot.pause(project_constants.PAUSE)
        plot.savefig('figures/{}_silhouette_for_{}_clusters'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower(), k))

def plot_elbow_graph(name, scores, k_values):
    plot.clf()
    plot.title('Elbow graph for {}'.format(name))
    plot.ylabel('score')
    plot.xlabel('# of clusters')
    plot.plot(k_values, scores, marker='.')
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_elbow_graph'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))

def plot_homogeneity_and_completeness(name, homogeneity, completeness, k_values):
    plot.clf()
    plot.title('Homogeneity and completeness for {}'.format(name))
    plot.ylabel('score')
    plot.xlabel('# of clusters')
    plot.plot(k_values, homogeneity, label='homogeneity', marker='.', color='darkorange')
    plot.plot(k_values, completeness, label='completeness', marker='s', color='navy')
    plot.legend(loc='best')
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_homogeneity_completeness_graph'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))

def compare_clusterers(name, original, reduced, features, reduced_features, classes, min_k):
    percent_difference_all = []
    percent_difference_by_class = []
    k_values = []
    unique_classes = numpy.unique(classes)
    for _ in unique_classes:
        percent_difference_by_class.append([])
    for c in range(0, len(original)):
        k_values.append(c + min_k)
        o_clusterer = original[c]
        r_clusterer = reduced[c]

        o_labels = o_clusterer.predict(features)
        r_labels = r_clusterer.predict(reduced_features)
        r_labels = map_labels(o_labels, r_labels, c + min_k)

        total_different = numpy.sum(o_labels != r_labels)
        percent_different = float(total_different) / len(features)
        percent_difference_all.append(percent_different)
        for i, u_class in enumerate(unique_classes):
            indices = numpy.nonzero(classes == u_class)
            total_different = numpy.sum(o_labels[indices] != r_labels[indices])
            percent_different = float(total_different) / len(indices[0])
            percent_difference_by_class[i].append(percent_different)
    plot.clf()
    plot.title('Percent different clusters with {}'.format(name))
    plot.ylabel('percent different clusters')
    plot.xlabel('# of clusters')
    plot.plot(k_values, percent_difference_all, label='total')
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_percent_difference_total_only'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))
    for i, u_class in enumerate(unique_classes):
        plot.plot(k_values, percent_difference_by_class[i], label='for {} class'.format(u_class))
    legend = plot.legend(loc='upper left', bbox_to_anchor=(1,1))
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_percent_difference_all'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()), bbox_extra_artists=(legend,), bbox_inches='tight')
    return percent_difference_all, k_values

def plot_total_difference(name, total_difference_by_dr):
    plot.clf()
    plot.title('Percent difference in clusters with {}'.format(name))
    plot.ylabel('percent different clusters')
    plot.xlabel('# of clusters')
    for i, dr in enumerate(total_difference_by_dr):
        color = COLORS[i]
        total_difference, k_values = total_difference_by_dr[dr]
        plot.plot(k_values, total_difference, label=dr, color=color)
    plot.legend(loc='best')
    plot.pause(project_constants.PAUSE)
    plot.savefig('figures/{}_percent_difference_total'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))

def map_labels(first_labels, second_labels, k):
    label_map = {}
    used_second_k = []
    new_second_labels = numpy.copy(second_labels)
    for c in range(0, k):
        c_first_indices = numpy.nonzero(first_labels == c)
        best_o_c = -1
        best_o_c_intersection = 0
        for o_c in range(0, k):
            if o_c in used_second_k:
                continue
            c_second_indices = numpy.nonzero(second_labels == o_c)
            intersection = numpy.intersect1d(c_first_indices, c_second_indices)
            if len(intersection) > best_o_c_intersection:
                best_o_c = o_c
                best_o_c_intersection = len(intersection)
        label_map[c] = best_o_c
        used_second_k.append(best_o_c)
        second_indices = numpy.nonzero(second_labels == best_o_c)
        new_second_labels[second_indices] = c
    return new_second_labels           
