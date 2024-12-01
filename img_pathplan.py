import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

# 画像からポテンシャルフィールドを復元
def load_potential_field_from_heatmap(image_path, cmap_name="jet", vmin=-1, vmax=1):

    # ヒートマップ画像を読み込み
    image = Image.open(image_path).convert("RGB")
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
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            color = tuple(image_data[x, y])
            field[x, y] = color_to_value[color]

    field = np.flipud(field)

    return field

def plan_path(potential_field, start, goal, map_size, step_size, tolerance):

    current_position = np.array(start, dtype=float)
    path = [start]
    velocity = np.array([0.0, 0.0])
    max_iterations = 1000  # 最大試行回数

    for _ in range(max_iterations):
        # ゴール判定
        if np.linalg.norm(current_position - np.array(goal)) < tolerance:
            print("ゴールに到達しました。")
            break

        # 現在位置の勾配を計算
        x, y = int(current_position[0]), int(current_position[1])
        grad_x = potential_field[min(x + 1, map_size - 1), y] - potential_field[max(x - 1, 0), y]
        grad_y = potential_field[x, min(y + 1, map_size - 1)] - potential_field[x, max(y - 1, 0)]
        gradient = np.array([grad_x, grad_y])

        # 速度と位置の更新
        acceleration = -gradient / np.linalg.norm(gradient + 1e-5)  # 正規化
        velocity = velocity * 0.9 + acceleration  # 慣性を加味
        velocity = velocity / np.linalg.norm(velocity + 1e-5) * step_size  # 正規化してステップサイズ適用
        current_position += velocity

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

def plot_potential_field(potential_field, start, goal, path=None):

    plt.figure(figsize=(10, 10))
    # plt.imshow(potential_field, cmap="coolwarm", origin="lower")
    plt.imshow(potential_field, cmap="jet", origin="lower")
    plt.colorbar(label="Potential Value")

    # 経路をプロット
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], marker='o', color='black', label='Path')

    plt.title("Potential Field with Start and Goal")
    plt.legend()
    plt.show()

if __name__=='__main__':

    # 環境マップの設定
    map_size = 50  # マップのサイズ
    obstacle_count = 5  # 小さめの障害物の数
    large_obstacle_count = 10 # 大きめの障害物の数
    large_obstacle_size = 3 # 大きめの障害物のサイズ
    safe_distance = 5

    # ポテンシャルフィールドの設定
    max_potential = 1.0
    min_potential = -1.0
    sigma = 0.5
    
    # ロボットの移動パラメータ
    step_size = 1.0  # ロボットのステップサイズ
    tolerance = 0.5  # ゴールに到達する許容距離

    start = (37, 35)
    goal = (34, 41)

    image_path = "pot_0.png" # 保存したヒートマップ画像のパス
    recovered_potential_field = load_potential_field_from_heatmap(image_path, vmin=min_potential, vmax=max_potential)

    # ポテンシャルフィールドを保存（テキスト形式）
    # path_w = 'pot2.txt'
    # with open(path_w, mode='w') as f:
    #     f.write(np.array2string(recovered_potential_field))

    path = plan_path(recovered_potential_field, start, goal, map_size, step_size, tolerance) # 経路計画
    print(path)
    plot_potential_field(recovered_potential_field, start, goal, path) # ヒートマップの表示と経路を可視化
