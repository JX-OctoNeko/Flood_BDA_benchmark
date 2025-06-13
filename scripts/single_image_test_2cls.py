import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score

def load_image(path, valid_values=None, mapping=None, merge_classes=None):
    """
    加载图像并检查是否为灰度图像，且灰度值是否在 valid_values 中。
    如果 mapping 不为 None，则将图像的灰度值根据 mapping 进行映射。
    如果 merge_classes 不为 None，则将指定类别合并为一个类别。
    如果不符合条件，抛出异常。
    """
    image = io.imread(path)
    
    # 检查是否为灰度图像
    if len(image.shape) > 2:
        raise ValueError(f"图像 {path} 不是灰度图像！请确保输入为单通道图像。")
    
    # 如果提供了 mapping，则进行灰度值映射
    if mapping is not None:
        image = np.vectorize(mapping.get)(image)  # 使用映射表转换灰度值
    
    # 如果需要合并类别
    if merge_classes is not None:
        for original_class, target_class in merge_classes.items():
            image[image == original_class] = target_class
    
    # 检查灰度值是否在 valid_values 中
    if valid_values is not None:
        unique_values = np.unique(image)
        if not set(unique_values).issubset(valid_values):
            raise ValueError(f"图像 {path} 的灰度值不在合法范围内！合法值为: {valid_values}，实际值为: {unique_values}.")
    
    return image

def calculate_metrics(gt, pred, n_classes, ignore_class=None):
    """计算多类别问题的各项指标，可选择忽略某个类别"""
    # 展平数组
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # 如果需要忽略某个类别，则过滤掉该类别
    if ignore_class is not None:
        mask = gt_flat != ignore_class  # 创建掩码，排除 ignore_class
        gt_flat = gt_flat[mask]
        pred_flat = pred_flat[mask]
    
    # 计算混淆矩阵
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(n_classes))
    
    # 计算 Precision、Recall、F1（按类别）
    precision = precision_score(gt_flat, pred_flat, average=None, labels=range(n_classes))
    recall = recall_score(gt_flat, pred_flat, average=None, labels=range(n_classes))
    f1 = f1_score(gt_flat, pred_flat, average=None, labels=range(n_classes))
    
    # 计算 IoU（交并比）
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    
    # 计算 Kappa 系数
    kappa = cohen_kappa_score(gt_flat, pred_flat)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "F1": f1,
        "Kappa": kappa,
        "Confusion Matrix": cm
    }

def main(gt_path, pred_path):
    """主函数"""
    # 定义合法的灰度值
    valid_values = {0, 1, 2}  # 合并后只有 0, 1, 2
    n_classes = len(valid_values)  # 类别数量
    
    # 定义灰度值映射规则
    mapping = {
        0: 0,    # 0 -> 0
        50: 1,   # 50 -> 1
        100: 2,  # 100 -> 2
        150: 3,  # 150 -> 3
        200: 4   # 200 -> 4
    }
    
    # 定义类别合并规则
    merge_classes = {
        2: 2,  # 2 -> 2
        3: 2,  # 3 -> 2
        4: 2   # 4 -> 2
    }
    
    # 加载 Ground Truth 和模型输出
    gt = load_image(gt_path, valid_values=valid_values, mapping=None, merge_classes=merge_classes)
    pred = load_image(pred_path, valid_values=valid_values, mapping=mapping, merge_classes=merge_classes)
    
    # 检查图像尺寸是否一致
    if gt.shape != pred.shape:
        raise ValueError("Ground Truth 和模型输出的图像尺寸不一致！")
    
    # 计算指标，忽略类别 0
    metrics = calculate_metrics(gt, pred, n_classes=n_classes, ignore_class=0)
    
    # 打印结果
    print("评估结果（忽略类别 0）：")
    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:")
            print(value)
        elif isinstance(value, np.ndarray):  # 如果是数组，逐个格式化
            print(f"{metric}:")
            for i, v in enumerate(value):
                if i == 0:  # 跳过类别 0
                    continue
                print(f"  Class {i}: {v:.4f}")
        else:  # 如果是标量值，直接格式化
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    # 设置 Ground Truth 和模型输出的路径
    gt_path = "src/data/xView/Process2/test/label/midwest-flooding_00000411_disaster/midwest-flooding_00000411_disaster_9.png"  # 替换为你的 Ground Truth 路径
    pred_path = "exp/xview-100%/out/epoch_199/unet/midwest-flooding_00000411_disaster_9.png"  # 替换为你的模型输出路径
    
    # 运行评估
    main(gt_path, pred_path)