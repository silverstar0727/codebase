import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    다양한 앙상블 방법론을 제공하는 클래스
    
    지원하는 앙상블 방법:
    1. Simple Average (단순 평균)
    2. Weighted Average (가중 평균)
    3. Geometric Mean (기하 평균)
    4. Harmonic Mean (조화 평균)
    5. Rank Average (순위 기반 평균)
    6. Power Average (거듭제곱 평균)
    7. Confidence-based Weighted Average (신뢰도 기반 가중 평균)
    """
    
    def __init__(self, csv_files: List[str], weights: Optional[List[float]] = None):
        """
        Args:
            csv_files: CSV 파일 경로 리스트
            weights: 각 모델의 가중치 (weighted average용)
        """
        self.csv_files = csv_files
        self.weights = weights if weights else [1.0] * len(csv_files)
        self.predictions = []
        self.class_columns = None
        
        # 정규화된 가중치
        self.normalized_weights = np.array(self.weights) / np.sum(self.weights)
        
        self._load_predictions()
        self._validate_predictions()
    
    def _load_predictions(self):
        """모든 CSV 파일을 로드하고 검증"""
        print(f"Loading {len(self.csv_files)} prediction files...")
        
        for i, file_path in enumerate(self.csv_files):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path}: {df.shape}")
            
            # 첫 번째 파일에서 클래스 컬럼 정의
            if i == 0:
                self.class_columns = [col for col in df.columns if col != 'ID']
                self.ids = df['ID'].values
            
            self.predictions.append(df)
    
    def _validate_predictions(self):
        """예측 파일들의 일관성 검증"""
        print("Validating prediction files...")
        
        # 모든 파일이 같은 ID를 가지는지 확인
        for i, df in enumerate(self.predictions):
            if not np.array_equal(df['ID'].values, self.ids):
                raise ValueError(f"ID mismatch in file {i}: {self.csv_files[i]}")
            
            # 클래스 컬럼이 일치하는지 확인
            df_classes = [col for col in df.columns if col != 'ID']
            if df_classes != self.class_columns:
                # 일치하지 않는 칼럼 확인
                missing_classes = set(self.class_columns) - set(df_classes)
                extra_classes = set(df_classes) - set(self.class_columns)
                if not missing_classes and not extra_classes:
                    # 모든 클래스가 일치하지만 순서가 다를 경우
                    self.class_columns = sorted(self.class_columns)
                    df_classes = sorted(df_classes)
                    if df_classes != self.class_columns:
                        raise ValueError(f"Class columns order mismatch in file {i}: {self.csv_files[i]}")
                else:
                    raise ValueError(f"Class columns mismatch in file {i}: {self.csv_files[i]}")
        
        print(f"✓ All files validated successfully")
        print(f"✓ Number of samples: {len(self.ids)}")
        print(f"✓ Number of classes: {len(self.class_columns)}")
    
    def simple_average(self) -> pd.DataFrame:
        """단순 평균 앙상블"""
        print("Performing Simple Average ensemble...")
        
        # 모든 예측을 3D 배열로 변환 (n_models, n_samples, n_classes)
        prob_arrays = []
        for df in self.predictions:
            prob_arrays.append(df[self.class_columns].values)
        
        stacked_probs = np.stack(prob_arrays, axis=0)
        avg_probs = np.mean(stacked_probs, axis=0)
        
        return self._create_result_dataframe(avg_probs)
    
    def weighted_average(self, weights: Optional[List[float]] = None) -> pd.DataFrame:
        """가중 평균 앙상블"""
        if weights is None:
            weights = self.normalized_weights
        else:
            weights = np.array(weights) / np.sum(weights)
        
        print(f"Performing Weighted Average ensemble with weights: {weights}")
        
        prob_arrays = []
        for df in self.predictions:
            prob_arrays.append(df[self.class_columns].values)
        
        stacked_probs = np.stack(prob_arrays, axis=0)
        weighted_probs = np.average(stacked_probs, axis=0, weights=weights)
        
        return self._create_result_dataframe(weighted_probs)
    
    def geometric_mean(self) -> pd.DataFrame:
        """기하 평균 앙상블 (확률의 곱의 n제곱근)"""
        print("Performing Geometric Mean ensemble...")
        
        prob_arrays = []
        for df in self.predictions:
            # 0에 가까운 값들을 처리하기 위해 작은 값 추가
            probs = df[self.class_columns].values
            probs = np.clip(probs, 1e-8, 1.0)  # 0 방지
            prob_arrays.append(probs)
        
        stacked_probs = np.stack(prob_arrays, axis=0)
        geo_mean_probs = np.exp(np.mean(np.log(stacked_probs), axis=0))
        
        # 정규화
        geo_mean_probs = geo_mean_probs / np.sum(geo_mean_probs, axis=1, keepdims=True)
        
        return self._create_result_dataframe(geo_mean_probs)
    
    def harmonic_mean(self) -> pd.DataFrame:
        """조화 평균 앙상블"""
        print("Performing Harmonic Mean ensemble...")
        
        prob_arrays = []
        for df in self.predictions:
            probs = df[self.class_columns].values
            probs = np.clip(probs, 1e-8, 1.0)  # 0 방지
            prob_arrays.append(probs)
        
        stacked_probs = np.stack(prob_arrays, axis=0)
        harmonic_mean_probs = len(self.predictions) / np.sum(1.0 / stacked_probs, axis=0)
        
        # 정규화
        harmonic_mean_probs = harmonic_mean_probs / np.sum(harmonic_mean_probs, axis=1, keepdims=True)
        
        return self._create_result_dataframe(harmonic_mean_probs)
    
    def rank_average(self) -> pd.DataFrame:
        """순위 기반 평균 앙상블"""
        print("Performing Rank Average ensemble...")
        
        # 각 모델의 예측을 순위로 변환
        rank_arrays = []
        for df in self.predictions:
            probs = df[self.class_columns].values
            # 각 샘플에 대해 클래스별 순위 계산 (높은 확률 = 낮은 순위)
            ranks = np.argsort(np.argsort(-probs, axis=1), axis=1) + 1
            rank_arrays.append(ranks)
        
        # 순위의 평균 계산
        stacked_ranks = np.stack(rank_arrays, axis=0)
        avg_ranks = np.mean(stacked_ranks, axis=0)
        
        # 순위를 확률로 변환 (낮은 순위 = 높은 확률)
        max_rank = len(self.class_columns)
        rank_probs = (max_rank + 1 - avg_ranks) / np.sum(max_rank + 1 - avg_ranks, axis=1, keepdims=True)
        
        return self._create_result_dataframe(rank_probs)
    
    def power_average(self, power: float = 2.0) -> pd.DataFrame:
        """거듭제곱 평균 앙상블"""
        print(f"Performing Power Average ensemble with power={power}...")
        
        prob_arrays = []
        for df in self.predictions:
            prob_arrays.append(df[self.class_columns].values)
        
        stacked_probs = np.stack(prob_arrays, axis=0)
        
        # 거듭제곱 후 평균, 그 다음 역거듭제곱
        power_mean_probs = np.power(np.mean(np.power(stacked_probs, power), axis=0), 1.0/power)
        
        # 정규화
        power_mean_probs = power_mean_probs / np.sum(power_mean_probs, axis=1, keepdims=True)
        
        return self._create_result_dataframe(power_mean_probs)
    
    def confidence_weighted_average(self) -> pd.DataFrame:
        """신뢰도 기반 가중 평균 앙상블"""
        print("Performing Confidence-based Weighted Average ensemble...")
        
        prob_arrays = []
        confidences = []
        
        for df in self.predictions:
            probs = df[self.class_columns].values
            prob_arrays.append(probs)
            
            # 각 샘플의 신뢰도를 최대 확률값으로 계산
            confidence = np.max(probs, axis=1)
            confidences.append(confidence)
        
        stacked_probs = np.stack(prob_arrays, axis=0)  # (n_models, n_samples, n_classes)
        stacked_confidences = np.stack(confidences, axis=0)  # (n_models, n_samples)
        
        # 각 샘플별로 신뢰도 기반 가중치 계산
        weighted_probs = np.zeros_like(stacked_probs[0])  # (n_samples, n_classes)
        
        for i in range(len(self.ids)):  # 각 샘플에 대해
            sample_confidences = stacked_confidences[:, i]  # (n_models,)
            
            # 신뢰도가 0에 가까운 경우 처리
            if np.sum(sample_confidences) == 0:
                sample_weights = np.ones(len(self.predictions)) / len(self.predictions)
            else:
                sample_weights = sample_confidences / np.sum(sample_confidences)
            
            # 가중 평균 계산
            weighted_probs[i] = np.average(stacked_probs[:, i, :], axis=0, weights=sample_weights)
        
        return self._create_result_dataframe(weighted_probs)
    
    def _create_result_dataframe(self, probabilities: np.ndarray) -> pd.DataFrame:
        """결과 DataFrame 생성"""
        result_df = pd.DataFrame({
            'ID': self.ids,
            **{class_name: probabilities[:, i] for i, class_name in enumerate(self.class_columns)}
        })
        
        # ID를 첫 번째 컬럼으로, 나머지는 알파벳 순으로 정렬
        class_columns_sorted = sorted(self.class_columns)
        result_df = result_df[['ID'] + class_columns_sorted]
        
        return result_df
    
    def ensemble_all_methods(self, output_dir: str = "ensemble_results") -> Dict[str, pd.DataFrame]:
        """모든 앙상블 방법을 실행하고 결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        methods = {
            # 'simple_average': self.simple_average,
            # 'weighted_average': self.weighted_average,
            # 'geometric_mean': self.geometric_mean,
            # 'harmonic_mean': self.harmonic_mean,
            # 'rank_average': self.rank_average,
            # 'power_average': lambda: self.power_average(power=2.0),
            'confidence_weighted': self.confidence_weighted_average,
        }
        
        results = {}
        
        print(f"\n{'='*60}")
        print("ENSEMBLE RESULTS SUMMARY")
        print(f"{'='*60}")
        
        for method_name, method_func in methods.items():
            print(f"\n--- {method_name.upper().replace('_', ' ')} ---")
            
            try:
                result_df = method_func()
                results[method_name] = result_df
                
                # CSV 파일로 저장
                output_path = os.path.join(output_dir, f"ensemble_{method_name}.csv")
                result_df.to_csv(output_path, index=False)
                
                # 결과 요약 출력
                self._print_result_summary(result_df, method_name)
                print(f"✓ Saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ Error in {method_name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Ensemble completed! Results saved in '{output_dir}' directory")
        print(f"{'='*60}")
        
        return results
    
    def _print_result_summary(self, df: pd.DataFrame, method_name: str):
        """결과 요약 출력"""
        class_probs = df[self.class_columns].values
        
        # 기본 통계
        mean_confidence = np.mean(np.max(class_probs, axis=1))
        min_confidence = np.min(np.max(class_probs, axis=1))
        max_confidence = np.max(np.max(class_probs, axis=1))
        
        print(f"  Average confidence: {mean_confidence:.4f}")
        print(f"  Confidence range: [{min_confidence:.4f}, {max_confidence:.4f}]")
        
        # 예측 분포
        predictions = np.argmax(class_probs, axis=1)
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"  Prediction distribution: {len(unique)} classes predicted")


def find_prediction_files(directory: str = ".", pattern: str = "*predictions*.csv") -> List[str]:
    """예측 파일들을 자동으로 찾기"""
    files = glob.glob(os.path.join(directory, pattern))
    files.sort()  # 파일명 순으로 정렬
    return files


def main():
    """메인 실행 함수"""
    print("🚀 Car Classification Ensemble Predictor")
    print("=" * 50)
    
    # 방법 1: 자동으로 예측 파일 찾기
    prediction_files = find_prediction_files(pattern="*.csv")
    
    if not prediction_files:
        print("❌ No prediction files found!")
        print("Please ensure your prediction files contain 'predictions' in their names")
        return
    
    print(f"📁 Found {len(prediction_files)} prediction files:")
    for i, file in enumerate(prediction_files, 1):
        print(f"  {i}. {file}")
    
    # 방법 2: 수동으로 파일 지정 (필요시 주석 해제)
    # prediction_files = [
    #     # "1.csv",
    #     # "2.csv", 
    #     # "3.csv",
    #     # "4.csv",
    #     # "5.csv",
    #     "siglip.csv",
    #     "siglip1fold-1.csv",
    #     # "siglip2.csv",
    #     # "siglip3.csv"
    # ]
    
    # 가중치 설정 (선택사항)
    # 모델 성능에 따라 조정 가능
    weights = None  # 동등한 가중치
    # weights = [0.14, 0.14, 0.14, 0.14, 0.14, 0.30]  # 사용자 정의 가중치
    
    try:
        # 앙상블 예측기 초기화
        ensemble = EnsemblePredictor(prediction_files, weights=weights)
        
        # 모든 앙상블 방법 실행
        results = ensemble.ensemble_all_methods()
        
        # 추천 방법
        print("\n🎯 RECOMMENDED ENSEMBLE METHODS:")
        print("=" * 50)
        print("1. 🥇 Confidence-based Weighted Average - 각 샘플별로 가장 확신하는 모델에 더 높은 가중치")
        print("2. 🥈 Weighted Average - 모델 성능에 따른 가중치 적용")
        print("3. 🥉 Geometric Mean - 확률의 곱을 이용한 보수적 예측")
        print("4. Simple Average - 가장 기본적이고 안정적인 방법")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your prediction files and try again.")


if __name__ == "__main__":
    main()