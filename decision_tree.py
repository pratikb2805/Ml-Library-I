import numpy as np


class Decision_Tree:
    def __init__(self, data_in, data_out):
        try:
            self.data_in = data_in.tolist()
            self.data_out = data_out.tolist()  # THIS CONVERTS data_in AND y TO LIST IF IT  IS NOT
        except:
            self.data_in = data_in
            self.data_out = data_out
        self.m = len(data_in)
        self.j = len(data_in[0])
        self.Node = self.build_tree(self.data_in, self.y)  # THIS NODE IS ROOT NODE OF THE TREE

    def count(self, y):
        """
        THIS FUNCTION COUNTS THE OCCURANCES OF LABELS 
        AND RETURNS THE DICTIONARdata_out
        """
        li = {}
        for label in y:
            if label not in li:
                li[label] = 0
            li[label] += 1
        return li

    def gini_index(self, data_in, y):  # impurity a a given node between 1 and 0
        counts = self.count(y)
        impurity = 1
        for label in counts:
            prob_of_lbl = counts[label] / float(len(data_in))
            impurity -= prob_of_lbl ** 2
        return impurity

    def gain(self, true_in, true_out, false_in, false_out, current_uncertainty):
        p = float(len(true_in)) / (len(true_in) + len(false_in))
        return current_uncertainty - p * self.gini(true_in, true_out) - (1 - p) * self.gini(false_in, false_out)

    class Question:
        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            val = example[self.column]
            if isinstance(val, int) or isinstance(val, float):
                return val >= self.value
            else:
                return val == self.value

    def partition(self, data_in, y, question):
        """
        THIS FUNCTION PARTITIONS THE DATA INTO TWO BRANCHS ACCORDING TO A GIVEN CONDITION
        AND RETURNS THE BRANCHES
        """
        true_in, true_out, false_in, false_out = [], [], [], []
        for i in range(len(data_in)):
            if question.match(data_in[i]):
                true_in.append(data_in[i])
                true_out.append(y[i])
            else:
                false_in.append(data_in[i])
                false_out.append(y[i])
        return true_in, true_out, false_in, false_out

    def find_best_split(self, data_in, y):
        """
        IT FINDS THE BEST QUESTION TO BE ASKED TO HAVE MAdata_inIMUM INFORMATION GAIN
        """
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(data_in, y)
        for col in range(self.n):
            values = set([row[col] for row in data_in])
            for val in values:
                question = self.Question(col, val)
                true_in, true_out, false_in, false_out = self.partition(data_in, y, question)
                if (len(true_in) == 0 or len(false_in) == 0):
                    continue
                gain = self.info_gain(true_in, true_out, false_in, false_out, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    class Leaf:
        """
        IT IS THE LEAF NODE THAT CONTAINS THE MOST CLASSIFIED INFO
        """

        def __init__(self, data_in, y):
            counts = Decision_Tree().class_counts(y)
            total = sum(counts.values())
            for label in counts.keys():
                counts[label] = str(counts[label] / total * 100) + "%"
            self.predictions = counts

    class Decision_Node:
        """
        IT IS THE NODE FROM WHICH BRANCHING OCCURS
        """

        def __init__(self, question, true_branch, false_branch):
            self.true_branch = true_branch
            self.false_branch = false_branch
            self.question = question

    def build_tree(self, data_in, y):  # recursion based function ..
        gain, question = self.find_best_split(data_in, y)  # keeps splitting nodes until pure data is obtained
        if gain == 0:
            return self.Leaf(data_in, y)
        true_in, true_out, false_in, false_out = self.partition(data_in, y, question)
        true_branch = self.build_tree(true_in, true_out)
        false_branch = self.build_tree(false_in, false_out)
        return self.Decision_Node(question, true_branch, false_branch)

    def classify(self, Node, example):
        if isinstance(Node, self.Leaf):
            return Node.predictions
        else:
            if (Node.question.match(example)):
                return self.classify(Node.true_branch, example)
            else:
                return self.classify(Node.false_branch, example)

    def predict(self, data_in_test):  # _________PREDICTS THE OUTPUT________#
        y_pred = []
        for example in data_in_test:
            d = self.classify(self.Node, example)
            v = list(d.values())
            k = list(d.keys())
            y_pred.append(k[v.index(max(v))])
        return np.array(y_pred)

    def accuracy(self, data_in_test, y_test):  # _________TESTS THE ACCURACdata_out OF THE MODEL______#
        y_pred = self.predict(data_in_test)
        a = np.array(y_pred == y_test)
        acc = np.mean(a) * 100
        return acc

    def predict_prob(self, data_in_test):
        y_pred = []
        for example in data_in_test:
            y_pred.append(self.classify(self.Node, example))
        return y_pred
