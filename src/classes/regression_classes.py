import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


class SimpleLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.w = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series) -> "SimpleLinearRegression":
        """
        фукнкция обучения - вычисляет параметры модели (веса) по данной выборке

        Inputs:
        X - матрица признаков
        y - вектор ответов

        Outputs:
        self - модель
        """
        X = np.asarray(X)
        X = X if not self.fit_intercept else self._add_intercept(X)
        y = np.array(y).flatten()

        if not np.isclose(np.linalg.det(X.T @ X), 0.0):
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        else: 
            self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        функция предсказания - предсказывает ответы модели по данной выборке

        Inputs:
        X - матрица признаков

        Outputs:
        y_pred - предсказания
        """
        X = np.asarray(X)

        if self.fit_intercept:
            ones_col = np.ones((X.shape[0], 1))
            X = np.hstack([ones_col, X])

        y_pred = X @ self.w

        return y_pred

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        # добавляем вектор единиц в x, чтобы не занулить смещение w0 (байес)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack([ones_col, X])

        return X

    def get_weights(self) -> np.ndarray | None:
        ''' 
        фун-я возвращает веса модели
        '''
        return self.w


class SGDLinearRegression:
    def __init__(self, fit_intercept: bool = True, learning_rate: float = 0.01,
                 n_iter: int = 100, batch_size: int = 10, random_state: int = 21):

        self.fit_intercept = fit_intercept
        self.lr = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None
        self.loss_history = []
        self.rng = np.random.RandomState(random_state)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        # добавляем вектор единиц в x, чтобы не занулить смещение w0 (байес)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack([ones_col, X])

        return X

    def _gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Вычисление градиентов MSE loss для батча
        L = (1/n) * Σ(y_pred - y)^2
        ∇L = (2/n) * X.T @ (y_pred - y_true) = (2/n) * X.T @ (X @ w - y)
        """

        n = X_batch.shape[0]
        y_pred = X_batch @ self.w
        error = y_pred - y_batch

        gradient = (2/n) * X_batch.T @ error  # type: ignore
        return gradient, error

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series) -> "SGDLinearRegression":
        """
        Обучение модели методом стохастического градиентного спуска

        Параметры:
        -----------
        X : матрица признаков (n_samples, n_features)
        y : вектор целевых значений (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        X_train = X if not self.fit_intercept else self._add_intercept(
            X)
        y_train = np.array(y).flatten()

        n_samples, n_features = X_train.shape
        self.w = self.rng.randn(n_features) * 0.01

        for epoch in tqdm(range(self.n_iter), desc=f'Training {self.__class__.__name__} Model'):
            epoch_loss_sum = 0
            batch_count = 0

            indices = np.arange(n_samples)

            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                if len(batch_indices) == 0:
                    continue

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                grad, batch_loss = self._gradient(X_batch, y_batch)
                self.w = self.w - self.lr * grad

                epoch_loss_sum += np.mean(batch_loss**2)
                batch_count += 1

            if batch_count > 0:
                epoch_avg_loss = epoch_loss_sum / batch_count
                self.loss_history.append(epoch_avg_loss)
            else:
                self.loss_history.append(0.0)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        функция предсказания - предсказывает ответы модели по данной выборке

        Inputs:
        X - матрица признаков

        Outputs:
        y_pred - предсказания
        """
        X = np.asarray(X)

        if self.fit_intercept:
            X = self._add_intercept(X)

        return X @ self.w

    def get_weights(self) -> np.ndarray | None:
        return self.w


class ClassicalGDLinearRegression:
    def __init__(self, fit_intercept: bool = True, learning_rate: float = 0.01,
                 n_iter: int = 1000, random_state: int = 21):

        self.fit_intercept = fit_intercept
        self.lr = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.w = None
        self.loss_history = []
        self.rng = np.random.RandomState(random_state)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        # добавляем вектор единиц в x, чтобы не занулить смещение w0 (байес)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack([ones_col, X])

        return X

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Вычисление градиентов MSE loss для всей выборки
        L = (1/n) * Σ(y_pred - y)^2
        ∇L = (2/n) * X.T @ (y_pred - y_true) = (2/n) * X.T @ (X @ w - y)
        """

        n = X.shape[0]
        y_pred = X @ self.w
        error = y_pred - y

        gradient = (2/n) * X.T @ error  # type: ignore
        return gradient, error

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series) -> 'ClassicalGDLinearRegression':
        """
        Обучение модели с помощью классического градиентного спуска

        Параметры:
        -----------
        X : матрица признаков (n_samples, n_features)
        y : вектор целевых значений (n_samples,)
        """
        X_train = np.asarray(X)
        y_train = np.asarray(y).flatten()

        if self.fit_intercept:
            X_train = self._add_intercept(X_train)

        n_samples, n_features = X_train.shape
        self.w = self.rng.randn(n_features) * 0.01

        for epoch in tqdm(range(self.n_iter), desc=f'Training {self.__class__.__name__} Model'):

            grad, batch_loss = self._gradient(X_train, y_train)
            self.w = self.w - self.lr * grad

            loss = np.mean(batch_loss ** 2)
            self.loss_history.append(loss)

            # if loss < best_loss - eps:
            #     best_loss = loss
            # else:
            #     print(f"Ранняя остановка (эпоха {epoch}): loss стабилизировался на {loss:.4f}")
            #     break

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        функция предсказания - предсказывает ответы модели по данной выборке

        Inputs:
        X - матрица признаков

        Outputs:
        y_pred - предсказания
        """
        X = np.asarray(X)

        if self.fit_intercept:
            X = self._add_intercept(X)

        return X @ self.w

    def get_weights(self) -> np.ndarray | None:
        return self.w


class LassoLinearRegression(SGDLinearRegression):
    def __init__(self, fit_intercept: bool = True, learning_rate: float = 0.01, n_iter: int = 100,
                 batch_size: int = 10, random_state: int = 21, alpha: float = 0.01):
        super().__init__(fit_intercept, learning_rate, n_iter, batch_size, random_state)
        self.alpha = alpha

    def _gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Градиент с L1 регуляризацией (субградиент)
        ∇L = (2/n) * X.T @ (X @ w - y) + alpha * sign(w)
        """

        n = X_batch.shape[0]
        y_pred = X_batch @ self.w
        error = y_pred - y_batch

        mse_grad = (2/n) * X_batch.T @ error
        l1_grad = self.alpha * np.sign(self.w)  # type: ignore

        if self.fit_intercept:
            l1_grad[0] = 0  # смещение не штрафуем

        return mse_grad + l1_grad, error


class RidgeLinearRegression(SGDLinearRegression):
    def __init__(self, fit_intercept: bool = True, learning_rate: float = 0.01,
                 n_iter: int = 1000, batch_size: int = 10, random_state: int = 21, alpha: float = 0.01):
        super().__init__(fit_intercept, learning_rate, n_iter, batch_size, random_state)

        self.alpha = alpha

    def _gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Градиент с L2 регуляризацией
        ∇L = (2/n) * X.T @ (X @ w - y) + 2 * alpha * w
        """

        n = X_batch.shape[0]
        y_pred = X_batch @ self.w
        error = y_pred - y_batch

        mse_grad = (2/n) * (X_batch.T @ error)
        l2_grad = 2 * self.alpha * self.w  # type: ignore

        if self.fit_intercept:
            l2_grad[0] = 0  # смещение не штрафуем

        return mse_grad + l2_grad, error


class ElasticNetLinearRegression(SGDLinearRegression):
    def __init__(self, fit_intercept: bool = True, learning_rate: float = 0.01,
                 n_iter: int = 1000, batch_size: int = 10, random_state: int = 21,
                 alpha: float = 0.01, l1_ratio: float = 0.5):
        super().__init__(fit_intercept, learning_rate, n_iter, batch_size, random_state)
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def _gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Градиент ElasticNet
        ∇L = (2/n) * X.T @ (X @ w - y) + 
             alpha * l1_ratio * sign(w) + 
             alpha * (1 - l1_ratio) * w
        """
        n = X_batch.shape[0]
        y_pred = X_batch @ self.w
        error = y_pred - y_batch

        mse_grad = (2/n) * (X_batch.T @ error)

        l1_grad = self.alpha * self.l1_ratio * np.sign(self.w)  # type: ignore
        l2_grad = self.alpha * (1 - self.l1_ratio) * self.w  # type: ignore

        if self.fit_intercept:
            l1_grad[0] = 0
            l2_grad[0] = 0

        return mse_grad + l1_grad + l2_grad, error


def my_R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_mean)**2)

    if ss_tot == 0:
        return np.float16(0.0)

    return 1 - (ss_res / ss_tot)


def my_MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    return np.mean(np.abs(y_true - y_pred))


def my_MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    return np.mean((y_true - y_pred)**2)


def my_RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    return np.sqrt(np.mean((y_true - y_pred)**2))
