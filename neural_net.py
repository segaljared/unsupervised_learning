from sklearn.neural_network import MLPClassifier
import numpy
import sklearn.metrics
import matplotlib.pyplot as plot
import project_constants
import complexity_analysis
import time

def run_neural_net_analysis(name, features, classes, test_features, test_classes, hidden_units):
    sample_weight = None
    unique_classes, class_counts = numpy.unique(classes, return_counts=True)
    class_weights = numpy.max(class_counts) / class_counts
    sample_weight = classes
    for i in range(len(unique_classes)):
        class_value = unique_classes[i]
        class_weight = class_weights[i]
        sample_weight = numpy.where(classes == class_value, class_weight, sample_weight)

    nn = MLPClassifier((20, 15), max_iter=50000)
    start = time.time()
    nn.fit(features, classes)
    training_time = (time.time() - start) * 1000
    print('converged in {}'.format(nn.n_iter_))

    predicted_classes = nn.predict(features)
    print('F1 score: {}'.format(complexity_analysis.f1_accuracy(classes, predicted_classes)))
    print(sklearn.metrics.classification_report(classes, predicted_classes))
    plot_roc_curves(name, classes, nn.predict_proba(features))
    print(sklearn.metrics.confusion_matrix(classes, predicted_classes))

    predicted_classes = nn.predict(test_features)
    print(sklearn.metrics.classification_report(test_classes, predicted_classes))
    plot_roc_curves(name, test_classes, nn.predict_proba(test_features))
    plot.savefig('figures/{}_roc_curve'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower()))
    print(sklearn.metrics.confusion_matrix(test_classes, predicted_classes))
    prediction_time = get_prediction_time_in_ms(nn, test_features)
    save_neural_net_perfromance_data(name, test_classes, predicted_classes, training_time, prediction_time)
    return nn

def from_array(array_classes):
    classes = []
    for class_array in array_classes:
        classes.append(numpy.argmax(class_array))
    return classes

def to_array(classes, num_classes):
    array_classes = []
    for sample_class in classes:
        class_array = [0.0] * num_classes
        class_array[int(sample_class)] = 1.0
        array_classes.append(class_array)
    return array_classes

def save_neural_net_perfromance_data(name, test_classes, predicted_classes, training_time, prediction_time):
    save_name = 'data/{}_nn_performance.txt'.format(name.replace(' ', '_').replace('(', '_').replace(')', '_').lower())
    with open(save_name, 'w') as f:
        f.write(name)
        f.write('\n\nTraining time(ms): {}\n'.format(training_time))
        f.write('Prediction time(ms): {}'.format(prediction_time))
        f.write('\n\nClassification report:\n\n')
        f.write(sklearn.metrics.classification_report(test_classes, predicted_classes))
        f.write('\n\n\nConfusion matrix:\n\n')
        matrix = sklearn.metrics.confusion_matrix(test_classes, predicted_classes)
        for i, row in enumerate(matrix):
            f.write('{} |'.format(i).rjust(8))
            for value in row:
                f.write(str(value).center(10))
            f.write('\n')
        f.write(' ' * 8)
        f.write(('-' * 10) * matrix.shape[1])
        f.write('\n')
        f.write(' ' * 8)
        for i in range(0, matrix.shape[1]):
            f.write(str(i).center(10))

def plot_roc_curves(name, true_classes, predicted_probs):
        # based off of: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        plot.clf()
        false_positives = {}
        true_positives = {}
        roc_auc = {}
        matrix_true_classes = numpy.array(to_array(true_classes, len(numpy.unique(true_classes))))
        for i in range(0, 2):
            false_positives[i], true_positives[i], _ = sklearn.metrics.roc_curve(matrix_true_classes[:,i], predicted_probs[:,i])
            roc_auc[i] = sklearn.metrics.auc(false_positives[i], true_positives[i])

        plot.title('ROC Curve')
        plot.xlabel('false positive rate')
        plot.ylabel('true positive rate')
        colors = ['lightcoral', 'firebrick']
        plot.plot([0,1], [0, 1], ls='--', lw=2)
        for i in range(0, 2):
            plot.plot(false_positives[i], true_positives[i], color=colors[i], lw=2, label='class %d (area = %f)' % (i, roc_auc[i]))
        plot.legend(loc='best')
        plot.pause(project_constants.PAUSE)

def perform_complexity_analysis(training_features, training_classes, problem_name, hidden_layer_sizes):
    def create_neural_net_hidden_units(hidden_units):
        return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_units, max_iter=10000)
    
    def hidden_units_to_string(hidden_units):
        string = '{}'.format(hidden_units[0])
        for h in range(1, len(hidden_units)):
            string += '_{}'.format(hidden_units[h])
        return string

    hidden_units = complexity_analysis.run_complexity_analysis('Neural Net Hidden Units for %s' % problem_name,
                                                           'Hidden Units',
                                                           'Accuracy',
                                                           create_neural_net_hidden_units,
                                                           hidden_layer_sizes,
                                                           training_features,
                                                           training_classes,
                                                           complexity_analysis.f1_accuracy,
                                                           folds=5,
                                                           parameter_to_string=hidden_units_to_string)
    return hidden_units

def weight_updates(create_neural_net, features, classes, max_iterations=10000, folds=5):
    fold_indices = []
    fold_size = int(len(classes) / folds)
    fold_start = 0
    for i in range(0, folds):
        fold_end = min(fold_start + fold_size, len(features) - 1)
        fold_indices.append(range(fold_start, fold_end))
        fold_start = fold_end + 1

    training_sets = []
    training_classes = []
    validation_sets = []
    validation_classes = []
    neural_nets = []
    for i in range(0, folds):
        training_indices = []
        validation_indices = fold_indices[i]
        for j in range(0, folds):
            if i != j:
                training_indices.extend(fold_indices[j])
        training_indices = numpy.array(training_indices).astype(int)
        training_sets.append(features[training_indices])
        training_classes.append(classes[training_indices])
        validation_sets.append(features[validation_indices])
        validation_classes.append(classes[validation_indices])
        neural_nets.append(create_neural_net())

    for i, nn in enumerate(neural_nets):
        nn.partial_fit(training_sets[i], training_classes[i], numpy.unique(classes))

    plot.ion()

    training_scores_means = []
    training_scores_stds = []
    validation_scores_means = []
    validation_scores_stds = []
    weight_updates = []

    for update in range(0, max_iterations):
        print(update)
        training_scores = []
        validation_scores = []
        neural_nets[0].partial_fit(training_sets[0], training_classes[0])
        for i, nn in enumerate(neural_nets):
            nn.partial_fit(training_sets[i], training_classes[i])
            if update % 10 == 0 and update > 1:
                training_scores.append(complexity_analysis.f1_accuracy(nn, training_sets[i], training_classes[i]))
                validation_scores.append(complexity_analysis.f1_accuracy(nn, validation_sets[i], validation_classes[i]))
        if len(training_scores) > 0:
            weight_updates.append(update)
            training_scores_means = numpy.append(training_scores_means, numpy.mean(training_scores))
            training_scores_stds = numpy.append(training_scores_stds, numpy.std(training_scores))
            validation_scores_means = numpy.append(validation_scores_means, numpy.mean(validation_scores))
            validation_scores_stds = numpy.append(validation_scores_stds, numpy.std(validation_scores))
            plot_neural_net_weight_updates_curve(weight_updates, training_scores_means, training_scores_stds,
                                                 validation_scores_means, validation_scores_stds)


def plot_neural_net_weight_updates_curve(weight_updates, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std):
    plot.clf()
    plot.title('Accuracy vs weight updates')
    plot.xlabel('Number of weight updates')
    plot.ylabel('Accuracy')
    plot.fill_between(weight_updates, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                      alpha=0.2, color="darkorange", lw=2)
    plot.plot(weight_updates, train_scores_mean, label="Training score", color="darkorange", lw=2)
    plot.fill_between(weight_updates, validation_scores_mean - validation_scores_std,
                      validation_scores_mean + validation_scores_std, alpha=0.2, color="navy", lw=2)
    plot.plot(weight_updates, validation_scores_mean, label="Validation score", color="navy", lw=2)
    plot.legend(loc="best")
    plot.grid()
    plot.pause(0.001)

def get_prediction_time_in_ms(nn, features):
    total_time = 0
    for _ in range(0, 100):
        start = time.time()
        nn.predict(features)
        total_time += time.time() - start
    # average time in s = time / 100; time in ms = time in s * 1000
    return total_time * 10
