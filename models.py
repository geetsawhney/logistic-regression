import numpy as np
import scipy.special as sc
import math


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LogisticRegression(Model):

    def __init__(self, gd_iterations, number_of_features_to_select, online_learning_rate):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.w = None
        self.online_learning_rate = online_learning_rate
        self.gd_iterations = gd_iterations
        self.number_of_features_to_select = number_of_features_to_select
        self.indices = None
        self.num_examples = None

    # def fit(self, X, y):
    #     # TODO: Write code to fit the model.
    #     self.num_input_features = X.shape[1]
    #     num_examples,num_input_features = X.shape;
    #     self.w=np.zeros([num_input_features],dtype=np.float)[np.newaxis]
    #     g = np.empty([num_examples], dtype=np.int)[np.newaxis]
    #     g_neg = np.empty([num_examples], dtype=np.int)[np.newaxis]
    #     ones=np.ones([num_examples])
    #     X_neg=-1*X;
    #     # print(g.shape)
    #
    #     for k in range(self.gd_iterations):
    #         for i,row in enumerate(X.toarray()):
    #             # print(row.shape)
    #             row=row[np.newaxis]
    #             val=self.w@row.T
    #             # print(val.shape)
    #             prob=sc.expit(val[0,0]);
    #             g[0,i]=y[i]*(1-prob);
    #             g_neg[0,i]=(1-y[i])*(prob);
    #
    #
    #
    #         # grad=(np.multiply(y,g_neg)@X)+(np.multiply(ones-y,g)@((-1)*X))
    #         # g=g[np.newaxis]
    #         # g_neg=g[np.newaxis]
    #         # print(X.shape)
    #         grad_1=g*X
    #         # print(grad_1.shape)
    #         grad_2=g_neg*X_neg
    #         delta = np.zeros(shape=(1, self.num_input_features), dtype=float);
    #         grad=grad_1+grad_2;
    #
    #         self.w= self.w + self.online_learning_rate*grad;

    def fit(self, X, y):
        # TODO: Write code to fit the model.

        self.num_examples, self.num_input_features = X.shape
        if self.number_of_features_to_select != -1:
            X = self.feature_Selection(X, y)

        self.num_examples, self.num_input_features = X.shape

        self.w = np.zeros(shape=(1, self.num_input_features), dtype=float)
        prob = np.zeros(shape=(1, self.num_examples), dtype=float)
        prob_neg = np.zeros(shape=(1, self.num_examples), dtype=float)

        for i in range(self.gd_iterations):
            delta = np.zeros(shape=(1, self.num_input_features), dtype=float)
            prob = sc.expit(self.w@X.T)
            prob_neg = sc.expit(-1 * (self.w@X.T))

            delta = (np.multiply(y, prob_neg)) * X + \
                (np.multiply(1 - y, prob)) * (-1 * X)
            self.w += self.online_learning_rate * delta

    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        try:
            if self.num_input_features != -1:
                X = X[:, self.indices]
        except:
            pass

        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        try:
            if self.num_input_features != -1:
                X = X[:, self.indices]
        except:
            pass

        y_hat = np.empty([num_examples], dtype=np.int)

        for i, row in enumerate(X.toarray()):
            row = row[np.newaxis]
            val = self.w@row.T
            prob = sc.expit(val)
            if prob >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat

    def calculate_conditional_entropy(self, prob_x1_y1, prob_x1_y0, prob_x0_y1, prob_x0_y0, px0, px1):
        log_prob__y1_x0 = log_prob__y1_x1 = log_prob__y0_x0 = log_prob__y0_x1 = 0

        if (prob_x0_y1 == 0):
            log_prob__y1_x0 = 0
        else:
            log_prob__y1_x0 = np.log2(prob_x0_y1)

        if (prob_x1_y1 == 0):
            log_prob__y1_x1 = 0
        else:
            log_prob__y1_x1 = np.log2(prob_x1_y1)

        if (prob_x1_y0 == 0):
            log_prob__y0_x1 = 0
        else:
            log_prob__y0_x1 = np.log2(prob_x1_y0)

        if (prob_x0_y0 == 0):
            log_prob__y0_x0 = 0
        else:
            log_prob__y0_x0 = np.log2(prob_x0_y0)

        entropy_yx1 = (prob_x1_y1 * log_prob__y1_x1) + \
            (prob_x1_y0 * log_prob__y0_x1)
        entropy_yx0 = (prob_x0_y1 * log_prob__y1_x0) + \
            (prob_x0_y0 * log_prob__y0_x0)
        entropy_yx = -((px0 * entropy_yx0) + (px1 * entropy_yx1))

        return entropy_yx

    # Calculate Information gain for each feature
    def feature_Selection(self, X, y):
        # Calculate  H(Y)
        unique_values_y, counts_y = np.unique(y, return_counts=True)
        prob_y0 = counts_y[0] / self.num_examples
        prob_y1 = counts_y[1] / self.num_examples
        # entropy_y = -((prob_y0)*np.log2(prob_y0)) -((prob_y1)*np.log2(prob_y1))
        ig = np.zeros([self.num_input_features])

        for feature in range(self.num_input_features):
            # flatten to 1 dimension
            temp = X[:, feature].toarray().squeeze()

            # count unique values in each column
            count_for_each_col = np.zeros([2])
            unique_values_xj, count_of_xj = np.unique(temp, return_counts=True)

            count_x1_y1 = count_x1_y0 = count_x0_y1 = count_x0_y0 = 0

            # if data is continuous we will use the mean as the threshold
            if len(unique_values_xj) != 2:
                mean = np.mean(temp)
                count_for_each_col[0] = np.where(temp < mean)[0].size
                count_for_each_col[1] = np.where(temp >= mean)[0].size
                # Get joint probabilities for x and y
                for row in range(self.num_examples):
                    if(temp[row] < mean):
                        if(y[row] == unique_values_y[0]):
                            count_x0_y0 += 1
                        elif(y[row] == unique_values_y[1]):
                            count_x0_y1 += 1
                    elif(temp[row] >= mean):
                        if(y[row] == unique_values_y[0]):
                            count_x1_y0 += 1
                        elif(y[row] == unique_values_y[1]):
                            count_x1_y1 += 1

            else:
                count_for_each_col[0] = count_of_xj[0]
                count_for_each_col[1] = count_of_xj[1]
                # Get joint probabilities for x and y
                for row in range(0, self.num_examples):
                    if(temp[row] == 0):
                        if(y[row] == unique_values_y[0]):
                            count_x0_y0 += 1
                        elif(y[row] == unique_values_y[1]):
                            count_x0_y1 += 1
                    elif(temp[row] == 1):
                        if(y[row] == unique_values_y[0]):
                            count_x1_y0 += 1
                        elif(y[row] == unique_values_y[1]):
                            count_x1_y1 += 1

            prob_xj_0 = count_for_each_col[0] / self.num_examples
            prob_xj_1 = count_for_each_col[1] / self.num_examples

            prob_x1_y1 = count_x1_y1 / count_for_each_col[1]
            prob_x1_y0 = count_x1_y0 / count_for_each_col[1]

            if(count_for_each_col[0] == 0):
                prob_x0_y0 = 0
                prob_x0_y1 = 0
            else:
                prob_x0_y1 = count_x0_y1 / count_for_each_col[0]
                prob_x0_y0 = count_x0_y0 / count_for_each_col[0]

            # calculate H(Y/X) for the given feature
            entropy_yx = self.calculate_conditional_entropy(
                prob_x1_y1, prob_x1_y0, prob_x0_y1, prob_x0_y0, prob_xj_0, prob_xj_1)
            ig[feature] = entropy_yx

        self.indices = (ig.argsort()[:self.number_of_features_to_select])
        new_X = X[:, self.indices]
        # print(self.indices)
        return new_X

# TODO: Add other Models as necessary.


class AdaBoost(Model):

    def __init__(self, iterations):
        super().__init__()
        self.iterations = iterations
        self.num_input_features = None
        self.h_t = []

    def fit(self, X, y):
        X = X.toarray()
        self.num_examples, self.num_input_features = X.shape
        y[y <= 0] = -1

        # Initialize c_array
        c_array=self.compute_c_values(X, y)
        # Initialize D
        D = np.ones(self.num_examples, dtype=np.float) / self.num_examples

        for t in range(self.iterations):
            h_tuple = self.compute_H_values(D, X, y,c_array)
            weight = h_tuple[3]
            if (weight < 0.000001):
                break

            c,j,ht,_, pred_vals= h_tuple

            x = X[:, j]
            alpha = (math.log((1 - weight) / weight)) / 2
            D = np.multiply(D, (np.exp(-alpha * np.multiply(y, pred_vals))))
            D = D / float(D.sum())
            self.h_t.append([c, j, ht, alpha])

    def compute_c_values(self, X, y):
        c_array = np.empty(
            (self.num_examples - 1, self.num_input_features), dtype=np.float)
        for j in range(self.num_input_features):
            x = sorted(X[:, j])
            for k in range(self.num_examples - 1):
                c_array[k, j] = (x[k + 1] + x[k]) / 2
        return c_array

    def compute_H_values(self, D, X, y, c_array):
        currError = np.inf
        best_direction = ()
        h_value=()
        for j in range(self.num_input_features):
            for k in range( self.num_examples - 1):
                c = c_array[k, j]
                if c == 0.:
                    continue
                for t in range(0, 2):
                    yhat_curr = np.ones(self.num_examples, dtype=np.int)
                    error_Array = np.ones(self.num_examples, dtype=np.float)
                    feature_array = X[:, j]
                    direction = (0, 0)
                    if (t == 0):
                        yhat_curr[feature_array  > c] = -1
                        direction = (-1, 1)
                    elif (t == 1):
                        yhat_curr[feature_array  <= c] = -1
                        direction = (1, -1)

                    error_Array[yhat_curr == y] = 0
                    weight = np.dot(error_Array, D)

                    if (weight < currError):
                        currError = weight
                        best_direction = direction
                        h_value = (c, j, direction, currError, yhat_curr)
        return h_value

    def predict(self, X):
        if self.h_t is [None]:
            raise Exception('fit must be called before predict.')

        num_examples, num_input_features = X.shape

        if num_input_features < self.num_input_features:
            # X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        X = X.toarray()

        weighted_sum = np.zeros(num_examples)
        yhat = np.empty(num_examples, dtype=np.int)

        for h in self.h_t:
            c,j,direction,alpha = h
            x = X[:, j]
            yhat = np.ones((num_examples), dtype=np.int)
            yhat[x > c] = direction[0]
            yhat[x <= c] = direction[1]

            weighted_sum = weighted_sum + (alpha * yhat)

        yhat = np.sign(weighted_sum)
        yhat[yhat < 0] = 0

        return yhat
