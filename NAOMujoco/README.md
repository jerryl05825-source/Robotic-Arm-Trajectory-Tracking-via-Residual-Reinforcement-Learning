# NAO V4 / MuJoCo

已导入 `../Nao V4/urdf/naoV40_generated_urdf/nao.urdf`，保持原始文件不变。
包含地面、灯光、初始站姿、25 个位置执行器以及 18 个关节联动约束。

## 启动

在本目录打开终端：

```bash
./run.sh
```

启动脚本自动使用本机已有的 `mujoco` Conda 环境。也可以直接运行：

```bash
conda activate mujoco
python simulate.py
```

默认固定躯干，适合先检查模型和后续手臂控制。鼠标左键旋转、右键平移、滚轮缩放。
查看器右侧 Control 面板可调整关节目标角度，关闭窗口退出。

```bash
./run.sh --floating                 # 浮动底座，尚无平衡/行走控制器
./run.sh --headless --seconds 10    # 无窗口仿真检查
./run.sh --rebuild                  # 从原始文件重新生成场景
MUJOCO_GL=egl ./run.sh --headless --screenshot preview.png
```

`run.sh` 将工作目录切换到本目录，截图相对路径以本目录为准。
其他机器安装 `requirements.txt`；可通过 `NAO_PYTHON=/path/to/python ./run.sh` 指定解释器。
无显示器时仅使用 `--headless`；离屏截图需要可用的 EGL 驱动。

## 右臂 CSV 轨迹播放

```bash
./run.sh --trajectory ../nao_right_arm_trajectory.csv --loop
./run.sh --trajectory ../nao_right_arm_trajectory.csv --headless
```

按 CSV 的 `time_s`、`RShoulderRoll_deg`、`RElbowRoll_deg` 进行关节角插值。
肩俯仰与肘旋转设为 0°，右臂水平；腕旋转为 +90°，手掌侧立；`RHand=0` 闭合手指。
通过带速度与重力偏置前馈的位置执行器进行动力学播放，保留 URDF 力矩上限。
初始状态直接放在轨迹起点。默认播放一次后保持终点；`--loop` 在终点停 1 秒、用 2 秒平滑返回、在起点停 1 秒后重播。
`--speed 0.5` 可半速播放。CSV 的二维手端坐标仅作参考，不用于三维逆解，真实 URDF 的肩肘偏移会使空间路径与简化二维模型不同。
本轨迹已验证完整播放：肩/肘最大角度跟踪误差小于 0.1°，无 MuJoCo 数值警告。

## 文件列表

- `simulate.py`：查看器、初始姿态和无窗口检查。
- `build_model.py`：URDF 转 MJCF，地面、控制器和 mimic 联动配置。
- `convert_meshes.py`：此数据集的 DAE 外观网格转换成 OBJ。
- `models/nao_v4.xml`：固定躯干场景。
- `models/nao_v4_floating.xml`：浮动底座场景。
- `models/visual_meshes/`：转换后的外观网格。
- `preview.png`：初始姿态预览。

## 当前范围

外观采用 DAE 转换网格和统一配色，未转换原始纹理；碰撞采用原始 STL 的凸包。
保留 URDF 网格缩放、关节范围、质量和惯量，并兼容 MuJoCo 惯量编译检查。
修复了源 URDF 中 `LAnklePitch/RAnklePitch` 与实际 `L_AnklePitch/R_AnklePitch` 文件名的差异。
为了先完成模型接入，当前只启用机器人与地面的碰撞，关闭机器人自碰撞。
关节使用基础位置伺服、URDF 力矩上限及数值稳定用的阻尼/转子惯量，尚未进行实机参数标定。
浮动底座可用于后续动力学开发，但不能将姿态伺服视作平衡控制器。

固定、浮动两场景均已完成 10 秒、2 ms 步长的无窗口仿真：状态有限、无 MuJoCo 警告；已离屏检查外观。

URDF 导入设置参考 [MuJoCo XML 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler)。
