import numpy as np
from optimizer import decent  # 前の回答で作成した更新関数

def train(X, Y, n2, epoch, epsilon):
    """
    学習の実行と重みの初期化を担当
    """
    n1 = len(X[0])
    n3 = len(Y[0])
    
    # --- 初期化 (Initialization) ---
    w2 = np.random.normal(0, 1, (n2, n1))
    w2 = np.insert(w2, 0, 0, axis=1) # バイアス項の追加
    w3 = np.random.normal(0, 1, (n3, n2))
    w3 = np.insert(w3, 0, 0, axis=1) # バイアス項の追加
    
    # --- 学習ループ (Training Loop) ---
    for _ in range(epoch):
        for x, y in zip(X, Y):
            # 1ステップの更新（optimizer内の関数を呼び出す）
            w = decent(x, y, w2, w3, epsilon)
            w2 = w['w2']
            w3 = w['w3']
            
    return dict(w2=w2, w3=w3)