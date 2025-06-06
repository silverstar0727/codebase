import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_high_res_confusion_matrix(
    val_csv_path,
    pred_csv_path,
    save_path="confusion_matrix_high_res.png",
    dpi=300,
    figsize=(40, 40),
    font_size=8,
    top_n_confused=50,
    show_full_matrix=False
):
    """
    Validation CSV와 Prediction CSV를 기반으로 고해상도 Confusion Matrix 생성
    
    Args:
        val_csv_path: validation CSV 파일 경로
        pred_csv_path: prediction CSV 파일 경로
        save_path: 저장할 이미지 파일 경로
        dpi: 이미지 해상도 (기본값: 300)
        figsize: 그림 크기 (기본값: (40, 40))
        font_size: 폰트 크기 (기본값: 8)
        top_n_confused: 상위 N개 혼동 클래스만 표시 (기본값: 50)
        show_full_matrix: 전체 매트릭스 표시 여부 (기본값: False)
    """
    
    # 1. Validation CSV 읽기
    print("Loading validation CSV...")
    val_df = pd.read_csv(val_csv_path)
    
    # 2. Prediction CSV 읽기
    print("Loading prediction CSV...")
    pred_df = pd.read_csv(pred_csv_path)
    
    # 3. ID를 기준으로 매칭
    print("Matching validation and prediction data...")
    
    # validation 파일에서 ID 추출 (filename에서 확장자 제거)
    val_df['ID'] = val_df['filename'].apply(lambda x: x.rsplit('.', 1)[0])
    
    # 공통 ID만 추출
    common_ids = set(val_df['ID'].values) & set(pred_df['ID'].values)
    print(f"Found {len(common_ids)} common samples")
    
    if len(common_ids) == 0:
        raise ValueError("No matching samples found between validation and prediction files")
    
    # 4. 매칭된 데이터 필터링
    val_filtered = val_df[val_df['ID'].isin(common_ids)].sort_values('ID')
    pred_filtered = pred_df[pred_df['ID'].isin(common_ids)].sort_values('ID')
    
    # 5. 예측 결과 생성
    print("Processing predictions...")
    
    # 예측 CSV에서 클래스 컬럼 추출 (ID 제외)
    class_columns = [col for col in pred_filtered.columns if col != 'ID']
    
    # 각 샘플에 대해 가장 높은 확률을 가진 클래스 선택
    pred_probs = pred_filtered[class_columns].values
    predicted_classes = np.argmax(pred_probs, axis=1)
    predicted_class_names = [class_columns[idx] for idx in predicted_classes]
    
    # 6. True labels와 Predicted labels 정렬
    true_labels = val_filtered['label'].values
    
    # 7. 클래스 이름을 인덱스로 매핑
    all_classes = sorted(list(set(true_labels) | set(predicted_class_names)))
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    
    true_indices = [class_to_idx[cls] for cls in true_labels]
    pred_indices = [class_to_idx[cls] for cls in predicted_class_names]
    
    print(f"Total classes: {len(all_classes)}")
    print(f"Samples: {len(true_indices)}")
    
    # 8. Confusion Matrix 생성
    print("Creating confusion matrix...")
    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(all_classes)))
    
    if show_full_matrix:
        # 전체 매트릭스 표시
        plot_confusion_matrix(
            cm, all_classes, save_path, dpi, figsize, font_size,
            title="Full Confusion Matrix"
        )
    else:
        # Top-N 혼동 클래스만 표시
        reduced_cm, reduced_classes = get_top_confused_classes(cm, all_classes, top_n_confused)
        plot_confusion_matrix(
            reduced_cm, reduced_classes, save_path, dpi, figsize, font_size,
            title=f"Top-{top_n_confused} Confused Classes"
        )
        
        # 추가로 전체 매트릭스도 저장 (숫자 표시 없이)
        full_save_path = save_path.replace('.png', '_full.png')
        plot_confusion_matrix(
            cm, all_classes, full_save_path, dpi//2, figsize, font_size//2,
            title="Full Confusion Matrix (No Annotations)"
        )
    
    # 9. 통계 정보 출력
    print_confusion_statistics(cm, all_classes, true_indices, pred_indices)

def get_top_confused_classes(cm, class_names, top_n):
    """가장 혼동이 많은 클래스들 추출"""
    print(f"Extracting top {top_n} confused classes...")
    
    # 대각선 제거 (정답 예측 제외)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # 각 클래스별 총 오분류 수 계산
    misclassified_counts = cm_no_diag.sum(axis=1)  # 각 true class의 오분류 수
    
    # 상위 N개 클래스 선택
    top_true_classes = np.argsort(misclassified_counts)[::-1][:top_n]
    
    # 이 클래스들과 관련된 모든 예측 클래스도 포함
    top_confused_classes = set(top_true_classes)
    for cls in top_true_classes:
        # 가장 많이 오분류된 예측 클래스들 추가
        pred_counts = cm_no_diag[cls]
        top_pred_classes = np.argsort(pred_counts)[::-1][:5]  # 상위 5개 예측 클래스
        top_confused_classes.update(top_pred_classes)
    
    # 정렬
    top_confused_classes = sorted(top_confused_classes)
    
    # 서브 매트릭스 추출
    reduced_cm = cm[np.ix_(top_confused_classes, top_confused_classes)]
    reduced_class_names = [class_names[i] for i in top_confused_classes]
    
    print(f"Selected {len(reduced_class_names)} classes for visualization")
    return reduced_cm, reduced_class_names

def plot_confusion_matrix(cm, class_names, save_path, dpi, figsize, font_size, title):
    """고해상도 Confusion Matrix 플롯 생성"""
    print(f"Creating {figsize[0]}x{figsize[1]} plot with {dpi} DPI...")
    
    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 고해상도 Figure 생성
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Heatmap 생성
    ax = sns.heatmap(
        cm,
        annot=True if len(class_names) <= 30 else False,  # 클래스가 많으면 숫자 표시 안함
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'shrink': 0.8}
    )
    
    # 제목 및 레이블 설정
    plt.title(title, fontsize=font_size * 2, pad=20)
    plt.xlabel("Predicted Label", fontsize=font_size * 1.5)
    plt.ylabel("True Label", fontsize=font_size * 1.5)
    
    # 틱 설정
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 고해상도로 저장
    print(f"Saving high resolution image to {save_path}...")
    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.2,
        facecolor='white',
        edgecolor='none'
    )
    
    plt.close()
    print(f"Confusion matrix saved successfully!")

def print_confusion_statistics(cm, class_names, true_indices, pred_indices):
    """혼동 행렬 통계 정보 출력"""
    print("\n" + "="*50)
    print("CONFUSION MATRIX STATISTICS")
    print("="*50)
    
    # 전체 정확도
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # 클래스별 정확도 (상위 10개 오분류)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    misclassified_counts = cm_no_diag.sum(axis=1)
    
    print(f"\nTop 10 Most Confused Classes:")
    top_confused_idx = np.argsort(misclassified_counts)[::-1][:10]
    
    for i, idx in enumerate(top_confused_idx):
        class_name = class_names[idx]
        total_samples = cm[idx].sum()
        correct_samples = cm[idx, idx]
        wrong_samples = misclassified_counts[idx]
        class_accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        print(f"{i+1:2d}. {class_name[:30]:<30} | "
              f"Accuracy: {class_accuracy:.3f} | "
              f"Correct: {correct_samples:3d} | "
              f"Wrong: {wrong_samples:3d} | "
              f"Total: {total_samples:3d}")

# 사용 예시
if __name__ == "__main__":
    # 예시 사용법
    val_csv_path = "val_files_fold2.csv"
    pred_csv_path = "predictions.csv"
    
    # 고해상도 confusion matrix 생성
    create_high_res_confusion_matrix(
        val_csv_path=val_csv_path,
        pred_csv_path=pred_csv_path,
        save_path="confusion_matrix_4K.png",
        dpi=400,  # 4K 해상도
        figsize=(50, 50),  # 큰 사이즈
        font_size=6,  # 작은 폰트
        top_n_confused=30,  # 상위 30개 혼동 클래스
        show_full_matrix=True  # 전체 매트릭스는 너무 크므로 False
    )
    
    print("\nFor full matrix (warning: very large file):")
    print("Set show_full_matrix=True and increase figsize to (100, 100)")