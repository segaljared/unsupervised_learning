import matplotlib.pyplot as plot
import numpy
import csv
import sklearn.model_selection as sklearn_m
import sklearn.metrics

def run_complexity_analysis(name, xlabel, ylabel, create_estimator_w_parameter, parameter_values, features, classes, scorer, parameter_to_string=None, folds=10, pause_at_end=False, save_csv=False, x_log_space=False):
    training_scores_means = []
    training_scores_stds = []
    validation_scores_means = []
    validation_scores_stds = []
    plot.ion()

    if parameter_to_string is not None:
        str_parameter_values = []
        for parameter_value in parameter_values:
            str_parameter_values.append(parameter_to_string(parameter_value))
    else:
        str_parameter_values = parameter_values

    fold_indices = []
    fold_size = int(len(classes) / folds)
    fold_start = 0
    for i in range(0, folds):
        fold_end = min(fold_start + fold_size, len(features) - 1)
        fold_indices.append(range(fold_start, fold_end))
        fold_start = fold_end + 1

    for iteration, parameter_value in enumerate(parameter_values):
        training_scores_for_iteration = []
        validation_scores_for_iteration = []
        # perform k folds cross validation:
        for i in range(0, folds):
            training_indices = []
            validation_indices = fold_indices[i]
            for j in range(0, folds):
                if i != j:
                    training_indices.extend(fold_indices[j])
            training_indices = numpy.array(training_indices).astype(int)
            # now we have our training and validation split of indices
            estimator = create_estimator_w_parameter(parameter_value)
            estimator.fit(features[training_indices], classes[training_indices])
            validation_scores_for_iteration.append(
                scorer(estimator, features[validation_indices], classes[validation_indices]))
            training_scores_for_iteration.append(
                scorer(estimator, features[training_indices], classes[training_indices]))
            print('Performed cross validation fold [%d/%d] for %s on parameter value %d of %d with score %f' % (
            i + 1, folds, name, iteration + 1, len(parameter_values), validation_scores_for_iteration[i]))
        training_scores_means = numpy.append(training_scores_means, numpy.mean(training_scores_for_iteration))
        training_scores_stds = numpy.append(training_scores_stds, numpy.std(training_scores_for_iteration))
        validation_scores_means = numpy.append(validation_scores_means, numpy.mean(validation_scores_for_iteration))
        validation_scores_stds = numpy.append(validation_scores_stds, numpy.std(validation_scores_for_iteration))
        # plot current
        if iteration > 0:
            plot_complexity_analysis(name, xlabel, ylabel, str_parameter_values[0:iteration + 1], training_scores_means,
                                     training_scores_stds, validation_scores_means, validation_scores_stds, x_log_space=x_log_space)

    if save_csv:
        save_data('%s_%s' % (name.replace(' ', '_').lower(), xlabel.replace(' ', '_')),
                  [('parameter_values', str_parameter_values),
                   ('training_means', training_scores_means),
                   ('training_stds', training_scores_stds),
                   ('validation_means', validation_scores_means),
                   ('validation_stds', validation_scores_stds)])

    plot_complexity_analysis(name, xlabel, ylabel, str_parameter_values, training_scores_means, training_scores_stds,
                             validation_scores_means, validation_scores_stds, x_log_space=x_log_space)
    plot.savefig('figures/%s_%s' % (name.replace(' ', '_').lower(), xlabel.replace(' ', '_')))
    if pause_at_end:
        plot.ioff()
        plot.show()
    best_param_index = numpy.argmax(validation_scores_means)
    return parameter_values[best_param_index]

def plot_complexity_analysis(name, xlabel, ylabel, parameter_values, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std, x_log_space=False):
    plot.clf()
    plot.title('Validation Curve for %s' % name)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.fill_between(parameter_values, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                      alpha=0.2, color='darkorange', lw=2)
    plot.plot(parameter_values, train_scores_mean, label='Training score', color='darkorange', lw=2)
    plot.fill_between(parameter_values, validation_scores_mean - validation_scores_std,
                      validation_scores_mean + validation_scores_std, alpha=0.2, color='navy', lw=2)
    plot.plot(parameter_values, validation_scores_mean, label='Validation score', color='navy', lw=2)
    if x_log_space:
        plot.xscale('log')
        print('log scale')
    plot.legend(loc='best')
    plot.pause(0.001)

def save_data(filename_w_o_extension, data):
    filename_to_try = 'data/%s.csv' % filename_w_o_extension
    count = 0
    column_names = []
    length = 0
    for column in data:
        column_names.append(column[0])
        if len(column[1]) > length:
            length = len(column[1])
    while True:
        try:
            with open(filename_to_try, 'x') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                for i in range(length):
                    row = []
                    for column in data:
                        if i < len(column[1]):
                            row.append(column[1][i])
                        else:
                            row.append('')
                    writer.writerow(row)
                return
        except OSError as err:
            if err.filename == filename_to_try:
                count += 1
                filename_to_try = 'data/%s_%d.csv' % (filename_w_o_extension, count)
            else:
                return
    
def f1_accuracy(estimator, X, y=None):
    if y is None:
        y = estimator
        predicted_classes = X
    else:
        predicted_classes = estimator.predict(X)
    if len(numpy.unique(y)) == 2:
        f1 = sklearn.metrics.f1_score(y, predicted_classes)
    else:
        f1 = sklearn.metrics.f1_score(y, predicted_classes, average='macro')
    return f1

def run_learning_curve_analysis(name, estimator, training_features, training_classes, folds=5):
    # base off of: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    training_sizes, training_scores, validation_scores = sklearn_m.learning_curve(estimator, training_features, training_classes, cv=folds)
    training_scores_mean = numpy.mean(training_scores, axis=1)
    training_scores_std = numpy.std(training_scores, axis=1)
    validation_scores_mean = numpy.mean(validation_scores, axis=1)
    validation_scores_std = numpy.std(validation_scores, axis=1)

    plot.clf()
    plot.title('%s Learning Curve' % name)
    plot.xlabel('Training examples')
    plot.ylabel('Accuracy')
    plot.grid()
    plot.fill_between(training_sizes, training_scores_mean - training_scores_std, training_scores_mean + training_scores_std,
                      alpha=0.2, color='darkorange', lw=2)
    plot.plot(training_sizes, training_scores_mean, label='Training score', color='darkorange', lw=2)
    plot.fill_between(training_sizes, validation_scores_mean - validation_scores_std,
                      validation_scores_mean + validation_scores_std, alpha=0.2, color='navy', lw=2)
    plot.plot(training_sizes, validation_scores_mean, label='Validation score', color='navy', lw=2)
    plot.legend(loc='best')
    #plot.ioff()
    #plot.show()
    plot.savefig('figures/%s_learning_curve' % name.replace(' ', '_').lower())
