import numpy as np


class PCA:
    def __init__(self, principal_components=None):
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance = None
        self.principal_components = principal_components

    def fit(self, X):
        M = np.mean(X, axis=0)
        C = X - M
        V = np.cov(C.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(V)
        self.explained_variance = np.diag(V)

    def transform(self, X):
        P = self.eigenvectors.T.dot(X.T)
        ag = np.flip(np.argsort(self.explained_variance))
        ag = ag[:self.principal_components]
        P = P.T
        P = P[:, ag]
        return P

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


np.random.seed(42)
num_cols = 10
X_train = np.random.normal(loc=0, scale=np.random.uniform(0.01, 6, num_cols), size=(20, num_cols))
X_test = np.random.normal(loc=5, scale=1, size=(5, num_cols))

pca = PCA(principal_components=2)
pca.fit(X_train)
X1 = pca.transform(X_train)
X2 = pca.transform(X_test)
s = sorted(pca.explained_variance, reverse=True)
sp = s / sum(s)
