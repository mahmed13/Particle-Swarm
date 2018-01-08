import pandas as pd
import math
import pickle
import argparse
import numpy as np
import random
import time

def _load_X(path):
    # Load the data.
    mat = np.loadtxt(path, dtype = int)
    max_doc_id = mat[:, 0].max()
    max_word_id = 10770
    X = np.zeros(shape = (max_doc_id, max_word_id))
    for (docid, wordid, count) in mat:
        X[docid - 1, wordid - 1] = count
    return X

def _load_train(data, labels):
    # Load the labels.
    y = np.loadtxt(labels, dtype = int)
    X = _load_X(data)

    # Return.
    return [X, y]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Particle Swarm",
        add_help = "How to use",
        prog = "python Swarm.py [train-data] [train-label] [test-data] <optional args>")
    parser.add_argument("paths", nargs = 3)
    parser.add_argument("-n", "--population", default = 200, type = int,
        help = "Population size [DEFAULT: 200].")
    parser.add_argument("-s", "--survival", default = 0.3, type = float,
        help = "Per-generation survival rate [DEFAULT: 0.3].")
    parser.add_argument("-m", "--mutation", default = 0.05, type = float,
        help = "Point mutation rate [DEFAULT: 0.05].")
    parser.add_argument("-g", "--generations", default = 50, type = int,
        help = "Number of generations to run [DEFAULT: 100].")
    parser.add_argument("-r", "--random", default = -1, type = int,
        help = "Random seed for debugging [DEFAULT: -1].")
    args = vars(parser.parse_args())

    # command line arguments
    if args['population'] > -1:
        population = args['population']

    if args['survival'] > -1:
        survival_rate = args['survival']

    if args['mutation'] > -1:
        mutation_rate = args['mutation']

    if args['generations'] > -1:
        max_number_generations = args['generations']

    if args['random'] > -1:
        np.random.seed(args['random'])

    # Read in the training data.
    X_train, Y_train = _load_train(args["paths"][0], args["paths"][1])
    #X_test, Y_test = _load_train(args["paths"][2], args["paths"][3])
    X_test = _load_X(args["paths"][2])

    Y_train = Y_train.reshape(-1,1)
    #Y_test = Y_test.reshape(-1,1)

    #remove zero columns
    train_sums = X_train.sum(axis=0)
    test_sums = X_test.sum(axis=0)
    to_delete = []
    for i in range(X_train.shape[1]):
        if train_sums[i] == 0 and test_sums[i] == 0:
            to_delete.append(i)
    X_train = np.delete(X_train, to_delete, 1)
    X_test = np.delete(X_test, to_delete, 1)


    # randomize order of data
    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    #X_test, Y_test = shuffle(X_test, Y_test, random_state=0)


    # add feature (sum)
    # sum_w = X_train.sum(axis=1).reshape(-1, 1)
    # X_train = np.hstack((X_train, sum_w))
    # sum_w = X_test.sum(axis =1).reshape(-1,1)
    # X_test = np.hstack((X_test, sum_w))

    # normalize sum column
    X_train = X_train / X_train.max(axis=1).reshape(-1,1)
    X_test = X_test / X_test.max(axis=1).reshape(-1,1)
    # X_train[-1] = X_train[-1] / X_train[-1].max(axis=0)
    # X_test[-1] = X_test[-1] / X_test[-1].max(axis=0)

    STEP_SIZE = 0.0001

    # compute result of sigmoid function from numpy array
    def sigmoid_function(x):
        y = 1. / (1. + (np.exp(-x)))
        return y

    # compute objective function from numpy array
    def objective_function(x):
        return np.sum(np.log(1 + np.exp(-x)))

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    ## change
    def log_likelihood(X, Y, b):
        Xb = np.dot(X, b)
        # handle -inf issue
        if np.max(Xb) > 400:
            return np.sum(Y*Xb - Xb)
        return np.sum(Y * Xb - np.log(1 + np.exp(Xb)))


    ## change
    def logistic_regression(X, Y, num_steps, step_size):

        # append dummy variable to X

        b_0 = np.ones((X.shape[0], 1))
        X = np.hstack((b_0, X))

        # initialize weights
        weights = np.ones(X.shape[1]).reshape(-1, 1)

        # step through gradient descent
        for step in range(num_steps):
            scores = np.dot(X, weights)
            predictions = sigmoid(scores).reshape(-1,1)
            # calculate error and gradient
            output_error_signal = Y - predictions
            gradient = np.dot(X.T, output_error_signal)

            #print('X', X.shape)
            #print('weight',weights.shape)
            #print('output', output_error_signal.shape)

            # update weights
            weights += step_size * gradient

            # print out objective value
            if step % math.floor(num_steps/5) == 0:
                pass
                print(log_likelihood(X, Y, weights))

        return weights

    def score(X, Y, w):
        final_scores = np.dot(X, w)
        # sigmoid function omitted for runtime error involving np.exp and large values
        final_scores[final_scores<=0] = 0
        final_scores[final_scores>0] = 1

        # if np.max(final_scores) > 4000:
        #     print('if&',np.max(final_scores), np.round(sigmoid(np.array([-50,-1,0,1,30,400,1000,4000]))))
        #pred = np.round(sigmoid(final_scores))
        #
        # #for i in range(len(pred)):
        #     #print(int(pred[i][0]))
        accuracy = (final_scores == Y).sum().astype(float) / len(final_scores)
        # #print('Accuracy: {0}'.format(accuracy))
        return accuracy

    def flip(p):
        return True if random.random() < p else False

    # saving pickles
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    # loading pickles
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # convert dense sparse matrix to normal matrix
    #X_train = convertSparse(X_train)
    #X_test = convertSparse(X_test)


    # account for dummy variable X*BetaT with intercept
    X_train = np.hstack((np.ones([X_train.shape[0],1]), X_train))

    X_test = np.hstack((np.ones([X_test.shape[0],1]), X_test))


    #weights = logistic_regression(X_train, Y_train, num_steps=50000, step_size=STEP_SIZE)
    #save_obj(weights,'weights')

    # weights_p = load_obj('weights')
    # print(X_test.shape, weights_p.shape)
    #
    # final_scores = np.dot(X_test, weights_p)
    # pred= np.round(sigmoid(final_scores))
    #
    # for i in range(len(pred)):
    #     print(int(pred[i][0]))
    # print('Accuracy: {0}'.format((pred == Y_test).sum().astype(float) / len(pred)))



    number_to_survive = math.ceil(population*survival_rate)
    # random initial weights
    #initial_weights = [np.random.uniform(low=-0.5, high=0.5, size=(X_train.shape[1],1)) for i in range(population)]
    initial_weights = np.array([4*np.random.randn(X_train.shape[1] ,1) for _ in range (population)])
    # parent's weights
    weights_p = []
    weights_p =  initial_weights

    mean = np.array(weights_p).mean(axis=0)
    min_indexes = []
    for i in range(max_number_generations):

        # using accuracy
        scores = np.array([score(X_train, Y_train, w) for w in weights_p])
        min_indexes = scores.argsort()[-number_to_survive:][::-1]
        # get parents who survived
        survived_parents = np.array([weights_p[i] for i in min_indexes])
        survived_parents_errors = np.array([scores[i] for i in min_indexes])
        #print(i, 'Accuracy:', np.mean(survived_parents_errors[0:5]), survived_parents_errors[0:5])

        # using objective values
        objective_values = np.array([log_likelihood(X_train, Y_train, w) for w in weights_p])
        objective_values = np.absolute(objective_values)
        # get indexes of minimum values
        min_indexes = objective_values.argsort()[:number_to_survive]
        # get parents who survived
        survived_parents = np.array([weights_p[i] for i in min_indexes])
        survived_parents_errors = np.array([objective_values[i] for i in min_indexes])
        #print(i, 'Objective values:', np.mean(survived_parents_errors[0:5]), survived_parents_errors[0:5])

        # calculate mean
        mean = np.array(weights_p).mean(axis=0)
        # calculate variance
        variance = np.array(weights_p).var(axis=0)
        # vectorized square root
        v_sqrt = np.vectorize(math.sqrt)
        # calculate standard deviation
        standard_dev = v_sqrt(variance)

        # randomly shuffle parents
        np.random.shuffle(survived_parents)

        children = []
        # create new population
        while len(children) < population:

            # mate parents to create each child
            for i in range(0, int(math.floor(len(survived_parents)/2))):

                # average parents to get children
                child = (survived_parents[i]+survived_parents[-i])/2.0

                # mutate child
                weights_to_mutate = [flip(mutation_rate) for _ in weights_p[0]]

                # draw new weight randomly from the gaussian distribution given mean and standard deviation
                child = [np.random.normal(mean[j], standard_dev[j]) if weights_to_mutate[j] else child[j] for j in range(len(mean))]

                children.append(child)
                if len(children) > population:
                    break

        weights_p = children

    # calculate all objective values
    objective_values = np.array([log_likelihood(X_train, Y_train, w) for w in weights_p])
    objective_values = np.absolute(objective_values)
    # get indexes of minimum values
    min_indexes = objective_values.argsort()[:number_to_survive]
    '''
    print('->',objective_values[min_indexes[0]])

    #weights_p = np.array(weights_p[min_indexes[0]]).reshape(-1,1)
    print('max',np.max(weights_p),'min',np.min(weights_p),'mean',np.mean(weights_p),'var',np.var(weights_p))
    top_accuracies = [score(X_test, Y_test, np.array(weights_p[min_indexes[i]]).reshape(-1,1)) for i in range(5)]
    print('Accuracies',top_accuracies)
    '''

    final_scores = np.dot(X_test, weights_p)
    # sigmoid function omitted for runtime error involving np.exp and large values
    final_scores[final_scores <= 0] = 0
    final_scores[final_scores > 0] = 1

    # if np.max(final_scores) > 4000:
    #     print('if&',np.max(final_scores), np.round(sigmoid(np.array([-50,-1,0,1,30,400,1000,4000]))))
    #pred = np.round(sigmoid(final_scores))
    pred = final_scores
    #
    for i in range(len(pred)):
        print(int(pred[i][0]))
    #accuracy = (final_scores == Y_test).sum().astype(float) / len(final_scores)
    #print('Accuracy: {0}'.format(accuracy))
