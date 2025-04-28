"""
Main evaluation script. Will produce numerous files in the destination folder. Among them:
- summary_metrics.csv for logistic regression evaluation
- summary_metrics_knn_1.csv for nearest neighbor classification evaluation
- cbir_mean_accuracy.csv for precision-at-k CBIR evaluation
- precision_recall_curve_cbir.jpg for the CBIR precision-recall curve 
"""
import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    pairwise_distances
)

from utils_cbir import compute_recall_at_k, plot_precision_recall_curve, retrieve_filenames
from utils_eval import load_vit_mae_model, load_dinov2_model, load_dino_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluation benchmark')
    parser.add_argument('--data_path', default='imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--pretrained_weights', default='https://huggingface.co/IverMartinsen/scampi-dino-vits16/resolve/main/vit_small_backbone.pth?download=true', type=str, help="Pretrained weights to evaluate. For ImageNet weights, use 'vit_mae' (ViT-MAE), 'dinov2' (DINOv2) or '' (DINOv1). For custom weights, use the path to the weights.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture. Options: vit_small, vit_base')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--destination', default='benchmark-results', type=str, help='Destination folder for storing the results.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--batch_size', default=128, type=int, help='GPU batch-size.')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============    
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # ============ building network ... ============
    if args.pretrained_weights == 'vit_mae':
        model = load_vit_mae_model(args)
    elif args.pretrained_weights == 'dinov2':
        model = load_dinov2_model(args)
    else:
        model = load_dino_model(args)
    
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    # ============ extract features ... ============
    print("Extracting features...")

    features = []
    labels = []

    for samples, labs in data_loader:
        if args.pretrained_weights == 'vit_mae':
            y = model.forward_features(samples)
        else:
            y = model(samples)
        features.append(y.detach().numpy())
        labels.append(labs.detach().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    class_names = ds.classes

    print("Features are ready!\nStart the classification.")
    
    for label in np.unique(labels):
        print(f"Class {label} has {np.sum(labels == label)} samples in the data set.")
    
    os.makedirs(args.destination, exist_ok=True)
    
    # ============ CBIR evaluation ... ============
    print("CBIR evaluation...")
    
    dists = pairwise_distances(features, features)
    dists += np.eye(len(labels)) * 1e12
    
    cbir_df = pd.DataFrame({"label": labels, "filename": filenames}) # DataFrame with CBIR metrics for each image 
    cbir_mean_df = pd.DataFrame() # DataFrame with mean CBIR metrics
    for k in ['k', 1] + [i for i in range(10, 200, 10)] + [500]:
        prec_at_k, rec_at_k = compute_recall_at_k(labels, dists, k)
        cbir_df[f"precision_at_{k}"] = prec_at_k
        cbir_df[f"recall_at_{k}"] = rec_at_k
        cbir_mean_df.loc["precision", k] = np.mean(prec_at_k)
        cbir_mean_df.loc["recall", k] = np.mean(rec_at_k)
    
    cbir_classwise_df = pd.DataFrame({ # DataFrame with mean CBIR metrics for each class
        "class": np.unique(labels),
        "precision_at_k": [cbir_df["precision_at_k"][cbir_df["label"] == k].mean() for k in np.unique(labels)]},
    )

    cbir_df.to_csv(os.path.join(args.destination, "cbir_accuracy.csv"))
    cbir_mean_df.to_csv(os.path.join(args.destination, "cbir_mean_accuracy.csv"))
    cbir_classwise_df.to_csv(os.path.join(args.destination, "cbir_classwise.csv"))
    
    # plot the precision-recall curve
    x = cbir_mean_df.loc["recall"].values
    y = cbir_mean_df.loc["precision"].values
    fname = os.path.join(args.destination, "precision_recall_curve_cbir.jpg")
    
    plot_precision_recall_curve(x, y, fname)
    
    print("CBIR evaluation done.")
    
    # ============ CBIR visualization ... ============
    
    # Plot the 10 images with best and worst precision-at-k
    top10 = cbir_df["filename"][cbir_df["precision_at_k"].argsort()[-10:][::-1]]
    bot10 = cbir_df["filename"][cbir_df["precision_at_k"].argsort()[:10]]

    def get_cname(fname):
        # util func for plotting
        lab = labels[np.where(np.array(filenames) == fname)[0][0]]
        return class_names[lab]

    def get_fpath(fname):
        # util func for plotting
        return os.path.join(args.data_path, get_cname(fname), fname)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    
    for i in range(10):
        query = top10.iloc[i]
        prec = cbir_df[cbir_df["filename"] == query][f"precision_at_k"].values[0]
        axes[i, 0].imshow(Image.open(get_fpath(query)).resize((224, 224)))
        axes[i, 0].set_title(get_cname(query) + " (query)" + f"\n(precision@k={prec:.2f})")
        axes[i, 0].axis("off")
        
        retrieved_filenames = retrieve_filenames(query, labels, filenames, dists)
        
        for j in range(1, 10):
            axes[i, j].imshow(Image.open(get_fpath(retrieved_filenames[j])).resize((224, 224)))
            axes[i, j].set_title(get_cname(retrieved_filenames[j]))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, "top10_retrieved_images.pdf"), bbox_inches="tight", dpi=300)
    plt.close()

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    
    for i in range(10):
        query = bot10.iloc[i]
        prec = cbir_df[cbir_df["filename"] == query][f"precision_at_k"].values[0]
        axes[i, 0].imshow(Image.open(get_fpath(query)).resize((224, 224)))
        axes[i, 0].set_title(get_cname(query) + " (query)" + f"\n(precision@k={prec:.2f})")
        axes[i, 0].axis("off")
        
        retrieved_filenames = retrieve_filenames(query, labels, filenames, dists)
        
        for j in range(1, 10):
            axes[i, j].imshow(Image.open(get_fpath(retrieved_filenames[j])).resize((224, 224)))
            axes[i, j].set_title(get_cname(retrieved_filenames[j]))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, "bot10_retrieved_images.pdf"), bbox_inches="tight", dpi=300)
    plt.close()

    # ============ Classification evaluation ... ============    
    summary_table = pd.DataFrame() # logistic regression metrics
    conf_mat_stats = {'preds': [], 'labels': []} # logistic regression confusion matrix stats
    summary_tables_knn = {k: pd.DataFrame() for k in args.nb_knn} # k-NN metrics
    class_wise_nn_stats = [] # class wise 1-NN metrics
    
    for seed in range(10):
        print(f"Evaluating seed {seed}...")
        
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
        
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # ============ logistic regression ... ============
        log_model = LogisticRegression(
            max_iter=10000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=seed,
        )

        log_model.fit(X_train, y_train)
        
        y_pred = log_model.predict(X_test)
        y_proba = log_model.predict_proba(X_test)
        
        conf_mat_stats['preds'].append(y_pred)
        conf_mat_stats['labels'].append(y_test)
            
        summary_table.loc[f"log_model_{seed}", "log_loss"] = log_loss(y_test, y_proba)
        summary_table.loc[f"log_model_{seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        summary_table.loc[f"log_model_{seed}", "accuracy"] = accuracy_score(y_test, y_pred)
        summary_table.loc[f"log_model_{seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
        
        # ============ k-NN ... ============
        for k in args.nb_knn:
            print(f"Evaluating k={k}...")
            
            knn = KNeighborsClassifier(n_neighbors=k, p=2)
            
            knn.fit(X_train, y_train)
            
            y_pred = knn.predict(X_test)
        
            # compute the accuracy and balanced accuracy and mean precision on the test set
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "accuracy"] = accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "f1_score"] = f1_score(y_test, y_pred, average="macro")

            # class wise recall, precision and f1 score
            if k == 1:
                df = pd.DataFrame({
                    "recall": recall_score(y_test, y_pred, average=None),
                    "precision": precision_score(y_test, y_pred, average=None),
                    "f1_score": f1_score(y_test, y_pred, average=None),
                })
                class_wise_nn_stats.append(df)
        
    # ============ summary ... ============
    summary_table.loc["mean", :] = summary_table.mean()
    summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))

    cm = confusion_matrix(np.concatenate(conf_mat_stats['labels']), np.concatenate(conf_mat_stats['preds']))
    cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    cm_display.plot(xticks_rotation="vertical", colorbar=False, text_kw={"fontsize": 7})
    cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)
    
    for k in args.nb_knn:
        summary_table_knn = summary_tables_knn[k]
        summary_table_knn.loc["mean", :] = summary_table_knn.mean() 
        summary_table_knn.to_csv(os.path.join(args.destination, f"summary_metrics_knn_{k}.csv"))
    
    # take the mean of the class wise statistics
    class_wise_nn_stats_mean = pd.concat(class_wise_nn_stats).groupby(level=0).mean()
    class_wise_nn_stats_mean.to_csv(os.path.join(args.destination, "class_wise_nn_stats.csv"))
    
    print("Summary metrics saved.")
