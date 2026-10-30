"""模型与训练结果保存工具

用法（在 notebook 中，训练完成且内核未重启时）:
    import save_utils
    save_utils.save_model_run(net, config, train_loss_list, val_loss_list,
                              num_train=80000, num_val=20000,
                              weights_root="weights")
"""

import os
import time
import torch
import matplotlib.pyplot as plt


def save_model_run(net, config, train_loss_list, val_loss_list,
                   num_train, num_val, weights_root="weights"):
    """把模型权重、配置、损失曲线和训练日志保存到带时间戳的子文件夹。

    参数:
        net: 训练好的网络
        config: 字典，包含模型与训练的超参数等复现信息
        train_loss_list: 每个 epoch 的训练 RMSPE（弧度）
        val_loss_list: 每个 epoch 的验证 RMSPE（弧度）
        num_train: 训练样本数
        num_val: 验证样本数
        weights_root: 保存的根目录（相对于当前工作目录）

    返回:
        保存文件夹的路径
    """
    # 1) 建立带时间戳的子文件夹（重名自动加 _1、_2 ...）
    time_str = time.strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(weights_root, "spstfv0_" + time_str)
    if os.path.exists(save_dir):
        idx = 1
        while os.path.exists(save_dir + "_" + str(idx)):
            idx = idx + 1
        save_dir = save_dir + "_" + str(idx)
    os.makedirs(save_dir, exist_ok=True)

    # 2) 模型权重与配置（torch 格式，加载用 torch.load）
    state = {"state_dict": net.state_dict(), "config": config}
    torch.save(state, os.path.join(save_dir, "model.pt"))

    # 3) 人类可读的配置
    config_lines = ["模型与训练配置", "保存时间: " + time.strftime("%Y-%m-%d %H:%M:%S")]
    for key in sorted(config.keys()):
        config_lines.append(str(key) + ": " + str(config[key]))
    with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(config_lines) + "\n")

    # 4) 损失曲线图（从损失列表重绘，只保存不显示）
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_loss_list) + 1), train_loss_list, "-", label="train loss")
    ax.plot(range(1, len(val_loss_list) + 1), val_loss_list, "m--", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSPE (rad)")
    ax.legend()
    ax.set_title("loss curve")
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=110)
    plt.close(fig)

    # 5) 训练日志（从损失列表重建，与训练时的打印一致）
    log_lines = ["训练样本: " + str(num_train) + "  验证样本: " + str(num_val)]
    for i in range(len(train_loss_list)):
        log_lines.append("epoch " + str(i + 1) +
                         "  train RMSPE: " + str(round(train_loss_list[i], 4)) +
                         "  val RMSPE: " + str(round(val_loss_list[i], 4)))
    with open(os.path.join(save_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("模型已保存到:", save_dir)
    return save_dir

def load_model_for_retrain(model_class, model_file, lr, device):
    """从保存的检查点加载模型用于继续训练，返回 (net, loss, updater)。

    参数:
        model_class: 网络类（例如 TransformerDOA），用于按保存时的配置重建结构
        model_file: 检查点文件夹路径（内含 model.pt），例如 "weights\\spstfv0_20261030_1110"
        lr: 继续训练的学习率
        device: torch 设备

    返回:
        (net, loss, updater)：已装入权重的网络、RMSPE 损失、Adam 优化器
    """
    from src.criterions import RMSPELoss
    checkpoint = torch.load(os.path.join(model_file, "model.pt"), map_location=device)
    config = checkpoint["config"]
    net = model_class(N=config["N"], T=config["T"], M=config["M"],
                      num_hiddens=config["num_hiddens"],
                      num_heads=config["num_heads"],
                      ffn_num_hiddens=config["ffn_num_hiddens"],
                      num_layers=config["num_layers"],
                      dropout=config["dropout"])
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    loss = RMSPELoss()
    updater = torch.optim.Adam(net.parameters(), lr=lr)
    return net, loss, updater
