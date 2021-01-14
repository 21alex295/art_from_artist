"""Random forest classifier."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC


class Classifier():
    """Wrap for the sklearn random forest classifier."""

    def __init__(self, algorithm,  params=[]):
        """Initialize randon forest parameters."""
        self._algorithm = algorithm
        if algorithm == "rf":
            if len(params) == 0:
                try:
                    self._classifier = RandomForestClassifier(verbose=1)
                except ValueError:
                    print("Classifier parameters are wrong")
                    exit(0)
            else:
                self._classifier = RandomForestClassifier(n_estimators=params[0],
                                                          max_depth=params[1],
                                                          random_state=0,
                                                          n_jobs=-1)
        elif algorithm == "svc":
            if len(params) == 0:
                self._classifier = SVC(verbose=1)
            else:
                try:
                    self._classifier = SVC(C=params[0],
                                           kernel=params[1],
                                           degree=params[2])
                except ValueError:
                    print("Classifier parameters are wrong")
                    exit(0)
        else:
            raise NameError(algorithm + " is not a valid algorithm")
        self.y_hat = []
        self.y = []

    def train(self, x_train, y_train):
        """Train random forest classifier with input data."""
        self._classifier.fit(x_train, y_train)

    def predict(self, x_test, y_test):
        """Test the classifier with the input data."""
        self.y_hat = self._classifier.predict(x_test)
        self.y = y_test
        return self.y_hat

    def conf_matrix(self):
        """Get confusion matrix from current data."""
        return confusion_matrix(self.y, self.y_hat)

    def accuracy(self):
        """Get accuracy for current data."""
        return accuracy_score(self.y, self.y_hat)

    def precision(self):
        """Get precision for current data."""
        return precision_score(self.y, self.y_hat, average="weighted")

    def feat_importance(self):
        """Get precision for current data."""
        if self._algorithm == "rf":
            return self._classifier.feature_importances_
        else:
            raise TypeError("SVM does not support feature importances.")
