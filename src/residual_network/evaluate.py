import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, classes_names, class_names, calc_score_fn):
    criterion = torch.nn.CrossEntropyLoss() # 損失関数の定義
    model.eval() # 推論モードに設定

    output_list = []
    target_list = []
    running_loss = 0.0

    with torch.no_grad(): # 勾配計算をオフにする
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device) # GPUへ転送
            outputs = model(inputs) # 予測
            loss = criterion(outputs, targets) # 損失計算

            # 30バッチごとに画像を1枚表示して予測確認
            if batch_idx % 30 == 0:
                plt.imshow(inputs[0, 1].cpu()) # 1チャンネル目を表示
                plt.gray()
                plt.show()
                # ImageNetラベル(1000クラス)での予測
                print(f"predict:{classes_names[outputs[0].argmax().item()]}")
                # CIFAR-10ラベル(10クラス)での正解
                print(f"correct:{class_names[targets[0].item()]}")

            # 集計用にリストへ追加
            output_list += [int(o.argmax()) for o in outputs]
            target_list += [int(t) for t in targets]
            running_loss += loss.item()

    # 最終的な精度と損失を計算
    test_acc, test_loss = calc_score_fn(output_list, target_list, running_loss, test_loader)
    print(f'test acc: {test_acc:<8}, test loss: {test_loss:<8}')