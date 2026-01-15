import numpy as np
from activations import relu, relu_derivative

def forward_propagation(x, w2, w3):
    """
    順伝播（Forward Propagation）
    $z_1$: 入力層, $z_2$: 隠れ層, $z_3$: 出力層
    """
    # 1. 入力層にバイアス(1)を挿入: z1 = [1, x1, x2, ...]^T
    z1 = np.insert(np.array([x]).T, 0, 1, axis=0)
    
    # 2. 隠れ層の総入力: u2 = w2 · z1
    u2 = np.dot(w2, z1)

    # 3. 隠れ層の出力（活性化関数）+ バイアス挿入: z2 = [1, f(u2)]^T
    z2 = np.insert(relu(u2), 0, 1, axis=0)
    
    # 4. 出力層の総入力: u3 = w3 · z2
    u3 = np.dot(w3, z2)

    # 5. 出力層の出力: z3 = u3
    z3 = u3
    
    return dict(z1=z1, z2=z2, z3=z3, u2=u2)

def backward_propagation(y, w2, w3, z1, z2, z3, u2):
    """
    逆伝播（Back Propagation / Error Backpropagation）
    $d_n$: 第n層の誤差（デルタ）, $dw_n$: 第n層の重みの勾配
    """
    # 1. 出力層の誤差（デルタ）: d3 = (z3 - t)
    # 誤差関数 E = 1/2 * (z3 - y)^2 の微分に対応
    d3 = (z3 - np.array([y]).T).T
    
    # 2. 隠れ層の誤差（デルタ）: d2 = (d3 · w3_without_bias) ⊙ f'(u2)
    # w3[:, 1:] はバイアス項の重みを除いた行列。⊙ は要素ごとの積。
    d2 = np.dot(d3, w3)[:, 1:] * relu_derivative(u2).T

    # 3. 出力層の重みの勾配: ∂E/∂w3 = d3^T · z2^T
    dw3 = d3.T * z2.T

    # 4. 隠れ層の重みの勾配: ∂E/∂w2 = d2^T · z1^T
    dw2 = d2.T * z1.T

    return dict(dw2=dw2, dw3=dw3)

def predict(x, w2, w3):
    """
    推論用の関数：最終的な予測値のみを返す
    """
    f = forward_propagation(x, w2, w3)
    return f['z3']