"""Random forest classifier."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


class RandomForest():
    """Wrap for the sklearn random forest classifier."""

    def __init__(self, nestimators=100, max_depth=None):
        """Initialize randon forest parameters."""
        self.test = "test"
        self._nestimators = nestimators
        self._max_depth = max_depth
        self._classifier = RandomForestClassifier(self._nestimators,
                                                  max_depth=self._max_depth,
                                                  random_state=0)
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
        return precision_score(self.y, self.y_hat)
