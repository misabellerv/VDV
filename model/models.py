from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

class ModelSelection:
    def __init__(self, config):
        self.seed = config.Training.Seed
        self.C = config.Models.SVM.C
        self.kernel = config.Models.SVM.Kernel
        self.max_depth = config.Models.DecisionTrees.MaxDepth
        self.min_samples_split = config.Models.DecisionTrees.MinSamplesSplit
        self.criterion = config.Models.DecisionTrees.Criterion
        self.use_label_encoder = config.Models.XGBoost.UseLabelEncoder
        self.n_estimators = config.Models.XGBoost.Nestimators
        self.learning_rate = config.Models.XGBoost.LearningRate
        self.eval_metric = config.Models.XGBoost.EvalMetric
        self.n_neighbors = config.Models.KNN.Neighbors
        self.metric = config.Models.KNN.Metric
        self.weights = config.Models.KNN.Weights
        self.class_weight = config.Models.ClassWeight

    def svm(self):
        print('Sucessefully imported SVM!')
        return  SVC(random_state=self.seed,
                        C=self.C,
                        kernel=self.kernel,
                        class_weight=self.class_weight,
                        random_state=self.seed
                    )

    def dt(self):
        print('Sucessefully imported Decision trees!')
        return DecisionTreeClassifier(random_state=self.seed,
                                          max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          criterion=self.criterion,
                                          class_weight=self.class_weight,
                                          random_state=self.seed
                                        )

    def xgb(self):
        print('Sucessefully imported XGB!')
        return XGBClassifier(use_label_encoder=self.use_label_encoder,
                                  eval_metric=self.eval_metric,
                                  n_estimators=self.n_estimators,
                                  learning_rate=self.learning_rate,
                                  random_state=self.seed
                                  )
    
    def knn(self):
        print('Sucessefully imported KNN!')
        return KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                    metric=self.metric,
                                    weights=self.weights
                                    )