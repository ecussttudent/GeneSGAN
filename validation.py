import os
import itertools
import numpy as np
import pandas as pd

def best_align_by_perm(y_true, y_pred):
    labels_true = sorted(pd.unique(y_true))
    labels_pred = sorted(pd.unique(y_pred))
    best_acc, best_map, best_pred = -1.0, {}, y_pred
    for perm in itertools.permutations(labels_true, len(labels_pred)):
        mapping = dict(zip(labels_pred, perm))
        y_pred_mapped = np.array([mapping.get(x, x) for x in y_pred])
        acc = (y_pred_mapped == y_true).mean()
        if acc > best_acc:
            best_acc, best_map, best_pred = acc, mapping, y_pred_mapped
    return best_pred, best_map, best_acc

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(base, "test_output", "clustering_result.csv")
    gt_path = os.path.join(base, "datasets", "toy_data_ground_truth.csv")
    out_dir = os.path.join(base, "test_output")
    os.makedirs(out_dir, exist_ok=True)

    df_pred = pd.read_csv(pred_path, usecols=["participant_id", "cluster_label"]).rename(columns={"cluster_label":"cluster_label_pred"})
    df_gt = pd.read_csv(gt_path).rename(columns={"cluster_label":"cluster_label_true"})
    df = df_gt.merge(df_pred, on="participant_id", how="inner")
    if df.empty:
        print("两份文件没有重叠的 participant_id")
        return

    y_true = df["cluster_label_true"].to_numpy()
    y_pred = df["cluster_label_pred"].to_numpy()

    y_pred_aligned, mapping, acc = best_align_by_perm(y_true, y_pred)
  
    cm = pd.crosstab(pd.Series(y_true, name="True"), pd.Series(y_pred_aligned, name="Pred"))
    cm_path = os.path.join(out_dir, "compare_confusion.csv")
    cm.to_csv(cm_path, index=True)

    print(f"样本数: {len(df)}")
    print(f"准确率(最佳标签对齐后): {acc:.4f}")
    print(f"标签映射(预测->真值): {mapping}")
    print(f"混淆矩阵已保存: {cm_path}")

if __name__ == "__main__":
    main()
