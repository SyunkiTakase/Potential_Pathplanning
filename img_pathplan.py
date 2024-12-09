import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

def load_potential_field_from_heatmap(image_path, map_size, cmap_name="jet", vmin=-1, vmax=1):
    """画像からポテンシャル値を復元"""

    # ヒートマップ画像を読み込み
    image = Image.open(image_path).convert("RGB")
    image = image.resize((map_size, map_size), Image.Resampling.LANCZOS) # ポテンシャルフィールドのヒートマップ画像をマップのサイズにリサイズ
    image_data = np.array(image)

    # カラーマップを取得
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # カラーマップの逆変換を準備
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)

    # ピクセル値をカラーマップのインデックスに変換
    normalized_values = scalar_mappable.to_rgba(np.linspace(vmin, vmax, 256))[:, :3]
    color_to_value = {tuple((normalized_values[i] * 255).astype(int)): vmin + i * (vmax - vmin) / 255 for i in range(256)}

    # ピクセルをスキャンしてポテンシャル値に変換
    field = np.zeros((image_data.shape[0], image_data.shape[1]))

    # キャッシュを用意して計算を効率化
    cache = {}

    def find_closest_color(color):
        if color in cache:
            return cache[color]
        closest_color = min(color_to_value.keys(), key=lambda c: np.sum((np.array(c) - np.array(color)) ** 2))
        cache[color] = color_to_value[closest_color]
        return cache[color]

    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            color = tuple(image_data[x, y])
            field[x, y] = find_closest_color(color)

    field = np.flipud(field)

    return field

def plan_path(potential_field, start, goal, map_size, step_size, tolerance):
    """ポテンシャルの値を元に経路を計画"""

    current_position = np.array(start, dtype=float) # 現在位置
    path = [start] # 経路
    velocity = np.array([0.0, 0.0]) # 速度
    max_iterations = 1000  # 最大試行回数

    for _ in range(max_iterations):
        # ゴール判定
        if np.linalg.norm(current_position - np.array(goal)) < tolerance:
            print("ゴールに到達しました。")
            break

        # 現在位置の勾配を計算
        x, y = int(current_position[0]), int(current_position[1]) # 現在位置のx,y座標
        grad_x = potential_field[min(x + 1, map_size - 1), y] - potential_field[max(x - 1, 0), y] # x方向の勾配
        grad_y = potential_field[x, min(y + 1, map_size - 1)] - potential_field[x, max(y - 1, 0)] # y方向の勾配
        gradient = np.array([grad_x, grad_y]) # 勾配

        # 速度と位置の更新
        acceleration = -gradient / np.linalg.norm(gradient + 1e-5) # 単位時間あたりの加速度
        velocity = velocity * 0.9 + acceleration # 単位時間あたりの速度
        velocity = velocity / np.linalg.norm(velocity + 1e-5) * step_size # 単位時間あたりの位置
        current_position += velocity # 移動後の位置を算出

        # 境界条件を適用
        current_position = np.clip(current_position, 0, map_size - 1)
        path.append(tuple(current_position.astype(int)))

        # 停止条件：速度が非常に小さい場合
        if np.linalg.norm(velocity) < 1e-2:
            print("速度が低下したため終了します。")
            break
    else:
        print("最大試行回数に達しました。")

    return path

def plot_pathplan(potential_field, start, goal, path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(potential_field, cmap="jet", origin="lower")
    plt.colorbar(label="Potential Value")
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], marker='o', color='black', label='Path')
    plt.scatter(start[1], start[0], color='green', label='Start', s=100, edgecolor='black')
    plt.scatter(goal[1], goal[0], color='red', label='Goal', s=100, edgecolor='black')
    plt.title("Potential Field with Path")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # パラメータ設定
    map_size = 50 # マップのサイズ
    step_size = 1.0 # ロボットのステップサイズ
    tolerance = 0.5 # ゴールに到達する許容距離
    max_potential = 1.0
    min_potential = -1.0

    # 開始位置とゴール位置
    start = (11, 18)
    goal = (38, 31)

    image_path = "pot_0.png"  # 保存されたヒートマップ画像
    recovered_potential_field = load_potential_field_from_heatmap(image_path, map_size, vmin=min_potential, vmax=max_potential)

    # 経路計画
    path = plan_path(recovered_potential_field, start, goal, map_size, step_size, tolerance)

    # プロット
    plot_pathplan(recovered_potential_field, start, goal, path)
