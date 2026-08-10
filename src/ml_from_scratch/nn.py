from abc import ABC,abstractmethod
import numpy as np

# 定义抽象基类
class Layer(ABC):
    num_in:int          # 输入维度
    num_out:int         # 输出维度
    use_bias:bool       # 是否加入偏置项

    W: np.ndarray
    b: np.ndarray | None
    X: np.ndarray
    y: np.ndarray

    grad_W: np.ndarray
    grad_b: np.ndarray | None

    rng: np.random.Generator    # 指定随机行为

    # 前向传播函数，根据输入x计算该层输出的y
    @abstractmethod
    def forward(self, X) -> np.ndarray:
        ...

    # 反向传播函数，根据上一层传回来的梯度，输出当前层累计的梯度
    @abstractmethod
    def backward(self, grad) -> np.ndarray:
        ...

    # 梯度更新函数，用于更新当前层的梯度
    @abstractmethod
    def update(self, learning_rate) -> None:
        ...


# 线性层-最基础的神经网络
class Linear(Layer):
    num_in:int
    num_out:int
    use_bias:bool

    W: np.ndarray
    b: np.ndarray | None
    X: np.ndarray
    y: np.ndarray

    grad_W: np.ndarray
    grad_b: np.ndarray | None

    rng: np.random.Generator

    def __init__(self, num_in, num_out, use_bias=True, rng=np.random.default_rng(seed=42)):
        self.num_in = num_in
        self.num_out = num_out
        self.use_bias = use_bias
        self.rng = rng
        # 参数的初始化
        self.W = rng.normal(0, 1.0, size=(num_in, num_out))
        if self.use_bias:
            self.b = np.zeros((1,num_out))
        else:
            self.b = None

    def forward(self ,X):
        # 前向传播：y = Wx + b
        self.X = X              # 数据x的维度为(batch_size, num_in)
        self.y = X@self.W       # 输出y的维度为(batch_size, num_out)
        if self.use_bias:
            self.y += self.b
        return self.y

    def backward(self, grad:np.ndarray):
        """
        反向传播，按照链式法则计算。上游梯度grad为(batch_size, num_out)。本层梯度要对batch_size取平均值。
        """
        # 上游grad的梯度为(batch_size, output)
        self.grad_W = (self.X.T) @ grad/grad.shape[0]           # grad_W与W的维度相同为(num_in, num_out)
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)  # grad_b与b的维度相同为(1,num_out)
        else:
            self.grad_b = None
        grad = grad @ (self.W).T    # 前向传播的grad的梯度为(batch_size, num_in)
        return grad
    