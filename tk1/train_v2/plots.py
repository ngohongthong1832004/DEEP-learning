# %% [markdown]
# ## ENHANCED VISUALIZATION & REPORTING

# %%
# ===== ADVANCED CHARTS =====

def plot_all_training_curves(all_histories):
    """Plot training curves for all models in a single figure"""
    
    n_models = len(CONFIG['models'])
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(20, 10))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model_name in enumerate(CONFIG['models']):
        ax = axes[idx]
        history = all_histories[model_name]
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        ax_loss = ax
        ax_loss.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax_loss.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax_loss.set_xlabel('Epoch', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10, color='b')
        ax_loss.tick_params(axis='y', labelcolor='b')
        ax_loss.grid(True, alpha=0.3)
        
        # Plot accuracy on secondary axis
        ax_acc = ax_loss.twinx()
        ax_acc.plot(epochs, history['train_acc'], 'g--', label='Train Acc', linewidth=2, alpha=0.7)
        ax_acc.plot(epochs, history['val_acc'], 'm--', label='Val Acc', linewidth=2, alpha=0.7)
        ax_acc.set_ylabel('Accuracy', fontsize=10, color='g')
        ax_acc.tick_params(axis='y', labelcolor='g')
        
        ax_loss.set_title(f'{model_name}', fontweight='bold', fontsize=11)
        
        # Combine legends
        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        ax_loss.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Training Curves - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "all_training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Training curves saved: {save_path}")

plot_all_training_curves(all_histories)

# %%
def plot_roc_curves(all_test_metrics):
    """Plot ROC curves for all models (One-vs-Rest for multiclass)"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    n_classes = len(class_names)
    
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        for model_idx, model_name in enumerate(models):
            metrics = all_test_metrics[model_name]
            y_true = metrics['y_true']
            y_probs = metrics['y_probs']
            
            # Binarize for current class
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_probs[:, class_idx]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[model_idx], linewidth=2,
                   label=f'{model_name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - {class_names[class_idx]}', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ROC Curves - All Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "roc_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ ROC curves saved: {save_path}")

plot_roc_curves(all_test_metrics)

# %%
def plot_precision_recall_curves(all_test_metrics):
    """Plot Precision-Recall curves for all models"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    n_classes = len(class_names)
    
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        for model_idx, model_name in enumerate(models):
            metrics = all_test_metrics[model_name]
            y_true = metrics['y_true']
            y_probs = metrics['y_probs']
            
            # Binarize for current class
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_probs[:, class_idx]
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            ap = average_precision_score(y_true_binary, y_score)
            
            ax.plot(recall, precision, color=colors[model_idx], linewidth=2,
                   label=f'{model_name} (AP={ap:.3f})')
        
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision-Recall - {class_names[class_idx]}', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Precision-Recall Curves - All Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "precision_recall_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Precision-Recall curves saved: {save_path}")

plot_precision_recall_curves(all_test_metrics)

# %%
def plot_model_performance_heatmap(all_test_metrics):
    """Create heatmap showing per-class F1 scores for all models"""
    from sklearn.metrics import precision_recall_fscore_support
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    # Calculate per-class F1 scores
    f1_matrix = []
    for model_name in models:
        metrics = all_test_metrics[model_name]
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        f1_matrix.append(f1)
    
    f1_matrix = np.array(f1_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=models,
                vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'})
    plt.title('Per-Class F1 Score Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "performance_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Performance heatmap saved: {save_path}")

plot_model_performance_heatmap(all_test_metrics)

# %%
def plot_model_characteristics():
    """Plot model size and inference time comparison"""
    
    models = CONFIG['models']
    
    # Get model parameters
    param_counts = []
    for model_name in models:
        num_classes = len(LABELS)
        model, total_params, _ = build_classifier(model_name, num_classes)
        param_counts.append(total_params / 1e6)  # In millions
        del model
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Parameter count comparison
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax1.barh(range(len(models)), param_counts, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('Parameters (Millions)', fontsize=11)
    ax1.set_title('Model Size Comparison', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, param) in enumerate(zip(bars, param_counts)):
        ax1.text(param + 0.5, bar.get_y() + bar.get_height()/2,
                f'{param:.2f}M', va='center', fontsize=10, fontweight='bold')
    
    # 2. Accuracy vs Size scatter
    ax2 = axes[1]
    test_accs = [all_test_metrics[m]['accuracy'] for m in models]
    
    scatter = ax2.scatter(param_counts, test_accs, c=range(len(models)), 
                         s=200, cmap='viridis', alpha=0.7, edgecolors='black')
    
    for i, model_name in enumerate(models):
        ax2.annotate(model_name.replace('_', ' '), 
                    (param_counts[i], test_accs[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    ax2.set_xlabel('Parameters (Millions)', fontsize=11)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title('Accuracy vs Model Size', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIRS["plots"], "model_characteristics.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Model characteristics saved: {save_path}")

plot_model_characteristics()

# %%
def plot_data_distribution():
    """Plot data distribution across splits"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    splits = [
        ('Train', train_df),
        ('Validation', val_df),
        ('Test', test_df)
    ]
    
    for ax, (split_name, df) in zip(axes, splits):
        counts = df['label_name'].value_counts().reindex(class_names, fill_value=0)
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        bars = ax.bar(range(len(class_names)), counts.values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'{split_name} Set\n(Total: {len(df)})', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Data Distribution Across Splits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "data_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Data distribution saved: {save_path}")

plot_data_distribution()

# %%
def plot_confusion_matrices_grid(all_test_metrics):
    """Plot all confusion matrices in a grid"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        cm = all_test_metrics[model_name]['confusion_matrix']
        acc = all_test_metrics[model_name]['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar=True, square=True)
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{model_name}\nAccuracy: {acc:.4f}', 
                    fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "confusion_matrices_grid.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Confusion matrices grid saved: {save_path}")

plot_confusion_matrices_grid(all_test_metrics)

# %%
def analyze_errors(all_test_metrics):
    """Analyze and visualize prediction errors"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Find best model
    best_model = max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])
    metrics = all_test_metrics[best_model]
    
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    y_probs = metrics['y_probs']
    
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        logging.info("No misclassified samples found!")
        return
    
    # Error analysis by class
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for class_idx in range(len(class_names)):
        ax = axes[class_idx]
        
        # Get errors where true label is current class
        class_mask = y_true == class_idx
        class_errors = misclassified_mask & class_mask
        
        if not class_errors.any():
            ax.text(0.5, 0.5, 'No Errors', ha='center', va='center', fontsize=14)
            ax.set_title(f'{class_names[class_idx]} - Errors', fontweight='bold')
            ax.axis('off')
            continue
        
        # Count predictions for misclassified samples
        error_predictions = y_pred[class_errors]
        error_counts = pd.Series(error_predictions).value_counts()
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(error_counts)))
        bars = ax.bar(range(len(error_counts)), error_counts.values, 
                     color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(error_counts)))
        ax.set_xticklabels([class_names[i] for i in error_counts.index], 
                          rotation=45, ha='right')
        ax.set_ylabel('Number of Errors', fontsize=10)
        ax.set_title(f'{class_names[class_idx]} Misclassified As...', 
                    fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, error_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Error Analysis - {best_model}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "error_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Error analysis saved: {save_path}")
    logging.info(f"Total misclassified: {len(misclassified_indices)} / {len(y_true)} ({len(misclassified_indices)/len(y_true)*100:.2f}%)")

analyze_errors(all_test_metrics)

# %% [markdown]
# ## COMPREHENSIVE REPORTS

# %%
def generate_html_report(all_test_metrics, all_histories):
    """Generate comprehensive HTML report"""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Disease Classification - Model Comparison Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .section h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        tr:hover {
            background-color: #f8f9ff;
        }
        .best {
            background-color: #d4edda;
            font-weight: bold;
        }
        .metric-card {
            display: inline-block;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .metric-card h4 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1em;
        }
        .metric-card p {
            font-size: 2em;
            font-weight: bold;
            color: #764ba2;
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .config-item {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .config-item strong {
            color: #764ba2;
        }
        .footer {
            background: #f8f9ff;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 2px solid #667eea;
        }
        .comparison-table {
            margin: 30px 0;
        }
        .rank {
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            font-weight: bold;
            color: white;
        }
        .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); }
        .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A8A8A8); }
        .rank-3 { background: linear-gradient(135deg, #CD7F32, #A0522D); }
        .rank-other { background: linear-gradient(135deg, #667eea, #764ba2); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌾 Rice Disease Classification</h1>
            <p>Multi-Model Comparison & Performance Report</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {timestamp}</p>
        </div>
        
        <div class="content">
"""
    
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = html_content.format(timestamp=timestamp)
    
    # Executive Summary
    html_content += """
            <div class="section">
                <h2>📊 Executive Summary</h2>
"""
    
    # Best model metrics
    best_model = max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])
    best_metrics = all_test_metrics[best_model]
    
    html_content += f"""
                <div style="text-align: center; margin: 30px 0;">
                    <div class="metric-card">
                        <h4>Best Model</h4>
                        <p>{best_model}</p>
                    </div>
                    <div class="metric-card">
                        <h4>Accuracy</h4>
                        <p>{best_metrics['accuracy']:.4f}</p>
                    </div>
                    <div class="metric-card">
                        <h4>F1 Score</h4>
                        <p>{best_metrics['f1']:.4f}</p>
                    </div>
                    <div class="metric-card">
                        <h4>Total Models</h4>
                        <p>{len(CONFIG['models'])}</p>
                    </div>
                </div>
"""
    
    # Configuration
    html_content += """
                <h3>Configuration</h3>
                <div class="config-grid">
"""
    
    config_items = {
        'Image Size': CONFIG['img_size'],
        'Batch Size': CONFIG['batch_size'],
        'Epochs': CONFIG['epochs'],
        'Learning Rate': CONFIG['lr'],
        'Use CBAM': CONFIG['use_cbam'],
        'Use Enhanced Head': CONFIG['use_better_head'],
        'Use Mixup': CONFIG['use_mixup'],
        'Use EMA': CONFIG['use_ema'],
        'Patience': CONFIG['patience'],
    }
    
    for key, value in config_items.items():
        html_content += f"""
                    <div class="config-item">
                        <strong>{key}:</strong> {value}
                    </div>
"""
    
    html_content += """
                </div>
            </div>
"""
    
    # Model Comparison
    html_content += """
            <div class="section">
                <h2>🏆 Model Performance Ranking</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Test Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>Val Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Sort models by accuracy
    models_sorted = sorted(CONFIG['models'], 
                          key=lambda m: all_test_metrics[m]['accuracy'], 
                          reverse=True)
    
    for rank, model_name in enumerate(models_sorted, 1):
        metrics = all_test_metrics[model_name]
        val_acc = all_best_val_accs[model_name]
        
        rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
        row_class = "best" if rank == 1 else ""
        
        html_content += f"""
                        <tr class="{row_class}">
                            <td><span class="rank {rank_class}">{rank}</span></td>
                            <td><strong>{model_name}</strong></td>
                            <td>{metrics['accuracy']:.4f}</td>
                            <td>{metrics['precision']:.4f}</td>
                            <td>{metrics['recall']:.4f}</td>
                            <td>{metrics['f1']:.4f}</td>
                            <td>{val_acc:.4f}</td>
                        </tr>
"""
    
    # Add ensemble if available
    if 'ensemble' in all_test_metrics:
        metrics = all_test_metrics['ensemble']
        html_content += f"""
                        <tr style="background-color: #fff3cd; font-weight: bold;">
                            <td>🎯</td>
                            <td><strong>ENSEMBLE</strong></td>
                            <td>{metrics['accuracy']:.4f}</td>
                            <td>{metrics['precision']:.4f}</td>
                            <td>{metrics['recall']:.4f}</td>
                            <td>{metrics['f1']:.4f}</td>
                            <td>-</td>
                        </tr>
"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
"""
    
    # Per-class performance
    html_content += """
            <div class="section">
                <h2>📈 Per-Class Performance</h2>
"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    for model_name in models_sorted[:3]:  # Top 3 models
        metrics = all_test_metrics[model_name]
        
        html_content += f"""
                <h3>{model_name}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        from sklearn.metrics import precision_recall_fscore_support
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        p, r, f1, s = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        
        for i, class_name in enumerate(class_names):
            html_content += f"""
                        <tr>
                            <td><strong>{class_name}</strong></td>
                            <td>{p[i]:.4f}</td>
                            <td>{r[i]:.4f}</td>
                            <td>{f1[i]:.4f}</td>
                            <td>{s[i]}</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
"""
    
    html_content += """
            </div>
"""
    
    # Dataset Information
    html_content += f"""
            <div class="section">
                <h2>📁 Dataset Information</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Split</th>
                            <th>Samples</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Training</strong></td>
                            <td>{len(train_df)}</td>
                            <td>{len(train_df)/len(collected_df)*100:.1f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Validation</strong></td>
                            <td>{len(val_df)}</td>
                            <td>{len(val_df)/len(collected_df)*100:.1f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Test</strong></td>
                            <td>{len(test_df)}</td>
                            <td>{len(test_df)/len(collected_df)*100:.1f}%</td>
                        </tr>
                        <tr style="background-color: #f0f0f0; font-weight: bold;">
                            <td>TOTAL</td>
                            <td>{len(collected_df)}</td>
                            <td>100%</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3>Class Distribution</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Train</th>
                            <th>Val</th>
                            <th>Test</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for class_name in class_names:
        train_count = len(train_df[train_df['label_name'] == class_name])
        val_count = len(val_df[val_df['label_name'] == class_name])
        test_count = len(test_df[test_df['label_name'] == class_name])
        total_count = train_count + val_count + test_count
        
        html_content += f"""
                        <tr>
                            <td><strong>{class_name}</strong></td>
                            <td>{train_count}</td>
                            <td>{val_count}</td>
                            <td>{test_count}</td>
                            <td>{total_count}</td>
                        </tr>
"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
"""
    
    # Close HTML
    html_content += """
        </div>
        
        <div class="footer">
            <p><strong>Rice Disease Classification System</strong></p>
            <p>Multi-Model Deep Learning Approach</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Classes: Brown Spot, Leaf Blast, Leaf Blight, Healthy
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    report_path = os.path.join(OUTPUT_DIRS["results"], "comprehensive_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"✓ HTML report saved: {report_path}")

generate_html_report(all_test_metrics, all_histories)

# %%
def generate_markdown_summary():
    """Generate markdown summary report"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    md_content = f"""# Rice Disease Classification - Model Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 Executive Summary

### Best Model: {max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])}

"""
    
    best_model = max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])
    best_metrics = all_test_metrics[best_model]
    
    md_content += f"""
| Metric | Value |
|--------|-------|
| **Accuracy** | {best_metrics['accuracy']:.4f} |
| **Precision** | {best_metrics['precision']:.4f} |
| **Recall** | {best_metrics['recall']:.4f} |
| **F1 Score** | {best_metrics['f1']:.4f} |

---

## 🏆 Model Performance Ranking

| Rank | Model | Test Acc | Precision | Recall | F1 Score | Val Acc |
|------|-------|----------|-----------|--------|----------|---------|
"""
    
    models_sorted = sorted(CONFIG['models'], 
                          key=lambda m: all_test_metrics[m]['accuracy'], 
                          reverse=True)
    
    for rank, model_name in enumerate(models_sorted, 1):
        metrics = all_test_metrics[model_name]
        val_acc = all_best_val_accs[model_name]
        
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        md_content += f"| {medal} {rank} | **{model_name}** | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {val_acc:.4f} |\n"
    
    if 'ensemble' in all_test_metrics:
        metrics = all_test_metrics['ensemble']
        md_content += f"| 🎯 | **ENSEMBLE** | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | - |\n"
    
    md_content += """
---

## 📈 Training Configuration

"""
    
    config_items = {
        'Image Size': CONFIG['img_size'],
        'Batch Size': CONFIG['batch_size'],
        'Epochs': CONFIG['epochs'],
        'Learning Rate': CONFIG['lr'],
        'Use CBAM Attention': CONFIG['use_cbam'],
        'Use Enhanced Head': CONFIG['use_better_head'],
        'Use Mixup Augmentation': CONFIG['use_mixup'],
        'Use EMA': CONFIG['use_ema'],
        'Patience (Early Stopping)': CONFIG['patience'],
    }
    
    for key, value in config_items.items():
        md_content += f"- **{key}:** {value}\n"
    
    md_content += f"""
---

## 📁 Dataset Information

### Data Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | {len(train_df)} | {len(train_df)/len(collected_df)*100:.1f}% |
| Validation | {len(val_df)} | {len(val_df)/len(collected_df)*100:.1f}% |
| Test | {len(test_df)} | {len(test_df)/len(collected_df)*100:.1f}% |
| **TOTAL** | **{len(collected_df)}** | **100%** |

### Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
"""
    
    for class_name in class_names:
        train_count = len(train_df[train_df['label_name'] == class_name])
        val_count = len(val_df[val_df['label_name'] == class_name])
        test_count = len(test_df[test_df['label_name'] == class_name])
        total_count = train_count + val_count + test_count
        
        md_content += f"| {class_name} | {train_count} | {val_count} | {test_count} | {total_count} |\n"
    
    md_content += """
---

## 🎯 Key Findings

"""
    
    # Find best and worst performing classes
    best_model_metrics = all_test_metrics[best_model]
    from sklearn.metrics import precision_recall_fscore_support
    
    p, r, f1, s = precision_recall_fscore_support(
        best_model_metrics['y_true'], 
        best_model_metrics['y_pred'], 
        labels=range(len(class_names)), 
        zero_division=0
    )
    
    best_class_idx = np.argmax(f1)
    worst_class_idx = np.argmin(f1)
    
    md_content += f"""
1. **Best Model:** {best_model} achieved {best_metrics['accuracy']:.4f} accuracy on test set
2. **Best Performing Class:** {class_names[best_class_idx]} (F1: {f1[best_class_idx]:.4f})
3. **Most Challenging Class:** {class_names[worst_class_idx]} (F1: {f1[worst_class_idx]:.4f})
4. **Total Models Trained:** {len(CONFIG['models'])}
"""
    
    if 'ensemble' in all_test_metrics:
        ensemble_acc = all_test_metrics['ensemble']['accuracy']
        improvement = (ensemble_acc - best_metrics['accuracy']) * 100
        md_content += f"5. **Ensemble Performance:** {ensemble_acc:.4f} ({'+' if improvement > 0 else ''}{improvement:.2f}% vs best single model)\n"
    
    md_content += """
---

## 📊 Visualization Files

All charts and plots have been saved to the `plots/` directory:

- `all_training_curves.png` - Training progress for all models
- `roc_curves.png` - ROC curves for each disease class
- `precision_recall_curves.png` - Precision-Recall curves
- `performance_heatmap.png` - Per-class F1 score heatmap
- `model_characteristics.png` - Model size and performance comparison
- `data_distribution.png` - Dataset distribution across splits
- `confusion_matrices_grid.png` - All confusion matrices
- `error_analysis.png` - Error distribution analysis
- `model_comparison.png` - Comprehensive comparison chart
- `per_class_comparison.png` - Per-class performance comparison

---

## 📝 Conclusion

This comprehensive analysis evaluated {len(CONFIG['models'])} lightweight deep learning models for rice disease classification. The best performing model, **{best_model}**, achieved an accuracy of **{best_metrics['accuracy']:.4f}** with excellent generalization across all disease classes.

**Recommendations:**
- Use **{best_model}** for deployment in production
- Consider ensemble approach for critical applications
- Focus on improving performance for **{class_names[worst_class_idx]}** class
- Continue monitoring model performance with new data

---

*Report generated automatically by Multi-Model Rice Disease Classification System*
"""
    
    # Save markdown report
    report_path = os.path.join(OUTPUT_DIRS["results"], "SUMMARY.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logging.info(f"✓ Markdown summary saved: {report_path}")

generate_markdown_summary()

# %%
logging.info("\n" + "="*80)
logging.info("ALL VISUALIZATIONS AND REPORTS GENERATED")
logging.info("="*80)
logging.info(f"\nOutput directory: {PATH_OUTPUT}")
logging.info(f"- Plots: {OUTPUT_DIRS['plots']}")
logging.info(f"- Results: {OUTPUT_DIRS['results']}")
logging.info(f"- Comparison: {OUTPUT_DIRS['comparison']}")
logging.info("="*80)