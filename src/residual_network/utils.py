import json
from pathlib import Path
from torchvision.datasets.utils import download_url
from sklearn.metrics import classification_report

# スコア（F1-scoreとLoss）の計算
def calc_score(output_list, target_list, running_loss, data_loader):
    # F1-scoreなどの統計情報を生成
    result = classification_report(output_list, target_list, output_dict=True)
    # 重み付き平均F1スコアを小数点6桁で取得
    acc = round(result['weighted avg']['f1-score'], 6)
    # 1サンプルあたりの損失を計算
    loss = round(running_loss / len(data_loader.dataset), 6)
    return acc, loss

# ImageNetのクラス名を取得
def get_imagenet_classes():
    file_path = Path("data/imagenet_class_index.json")
    if not file_path.exists():
        # クラス名対応表がない場合はダウンロード
        download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

    with open(file_path) as f:
        data = json.load(f)
    
    # リスト形式で、各要素が {"en": "..."} を持っている場合の抽出処理
    # 例: [{"num": "n01440764", "en": "tench", "ja": "テンチ"}, ...]
    class_names = [x["en"] for x in data]
    return class_names