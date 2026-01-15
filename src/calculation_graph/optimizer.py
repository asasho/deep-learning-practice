from model import forward_propagation, backward_propagation

def gradient_descent_step(x, y, w2, w3, epsilon):
    """
    1ステップ分の学習（パラメータ更新）を行う
    """
    # 1. 予測と誤差逆伝播
    f = forward_propagation(x, w2, w3)
    b = backward_propagation(y, w2, w3, f['z1'], f['z2'], f['z3'], f['u2'])
    
    # 2. 重みの更新
    w2_updated = w2 - epsilon * b['dw2']
    w3_updated = w3 - epsilon * b['dw3']
    
    return dict(w2=w2_updated, w3=w3_updated)