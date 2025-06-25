from Data_Preprocessing import keep_largest_island, run_total
from train import predict_2_5d_single_patient, build_2_5d_unet_model
import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import nibabel as nib
import gc
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_segmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 預測 左右腦室 全腦室  
# 從全腦室中去除左右腦室 利用keep_largest_island 取出三腦室
# 再去除三腦室 取出四腦室

def process_single_patient(dataset_path: str, output_dir: Optional[str] = None, skip_existing: bool = False) -> Dict[str, Union[str, bool, float]]:
    """
    處理單一病患的 DICOM 資料，進行完整的腦部分割流程
    
    Args:
        dataset_path (str): DICOM 資料夾路徑
        output_dir (str, optional): 輸出目錄，若為 None 則輸出到原始資料夾
        skip_existing (bool): 是否跳過已存在的結果檔案
        
    Returns:
        Dict: 處理結果，包含成功狀態、錯誤訊息、處理時間等
    """
    start_time = time.time()
    patient_id = Path(dataset_path).name
    result = {
        'patient_id': patient_id,
        'dataset_path': dataset_path,
        'success': False,
        'error_message': None,
        'processing_time': 0,
        'output_files': []
    }
    
    try:
        logger.info(f"開始處理病患: {patient_id}")
        
        # 確定輸出目錄
        if output_dir:
            output_path = Path(output_dir) / patient_id
            output_path.mkdir(parents=True, exist_ok=True)
            DATASET = str(output_path)
        else:
            DATASET = dataset_path
            
        # 檢查是否需要跳過已存在的結果
        if skip_existing:
            expected_files = ['CSF.nii.gz', 'Ventricles.nii.gz', 'Ventricle_L.nii.gz', 
                            'Ventricle_R.nii.gz', 'Third_Ventricle.nii.gz', 'Fourth_Ventricle.nii.gz']
            if all(Path(DATASET, f).exists() for f in expected_files):
                logger.info(f"跳過已存在的結果: {patient_id}")
                result['success'] = True
                result['processing_time'] = time.time() - start_time
                result['output_files'] = [str(Path(DATASET, f)) for f in expected_files]
                return result
        
        # 執行預處理
        logger.info(f"執行預處理: {patient_id}")
        run_total(dataset_path)
        
        # 如果有輸出目錄，複製 original.nii.gz
        if output_dir and dataset_path != DATASET:
            original_file = Path(dataset_path) / "original.nii.gz"
            if original_file.exists():
                import shutil
                shutil.copy2(original_file, Path(DATASET) / "original.nii.gz")
        
        # 建立模型
        logger.info(f"載入模型: {patient_id}")
        model = build_2_5d_unet_model((512, 512, 1))
        file_path = DATASET + "/original.nii.gz"
        
        # 定義模型路徑和結構名稱
        models_to_process = [
            ("model/CSF.keras", "CSF"),
            ("model/Ventricles.keras", "Ventricles"),
            ("model/Ventricle_L.keras", "Ventricle_L"),
            ("model/Ventricle_R.keras", "Ventricle_R")
        ]
        
        # 處理每個結構
        for model_path, structure_name in models_to_process:
            logger.info(f"預測 {structure_name}: {patient_id}")
            
            # 檢查模型檔案是否存在
            if not Path(model_path).exists():
                # 嘗試舊路徑
                old_model_path = f"train_2D/predict_argv/{structure_name}.keras"
                if Path(old_model_path).exists():
                    model_path = old_model_path
                else:
                    raise FileNotFoundError(f"找不到模型檔案: {model_path}")
            
            # 修復 .keras 檔案實際上是 H5 格式的問題
            try:
                model.load_weights(model_path)
            except:
                # 如果 .keras 檔案載入失敗，嘗試當作 H5 檔案載入
                import tempfile
                import shutil
                temp_h5_path = tempfile.mktemp(suffix='.h5')
                shutil.copy(model_path, temp_h5_path)
                model.load_weights(temp_h5_path)
                import os
                os.unlink(temp_h5_path)
            predict_2_5d_single_patient(model, file_path, False, name=structure_name)
            
            # 記錄輸出檔案
            output_file = f"{DATASET}/{structure_name}.nii.gz"
            if Path(output_file).exists():
                result['output_files'].append(output_file)
        
        # 計算第三腦室和第四腦室
        logger.info(f"計算第三和第四腦室: {patient_id}")
        
        ventricles = nib.load(f'{DATASET}/Ventricles.nii.gz')
        ventricles_data = ventricles.get_fdata()
        
        ventricle_L = nib.load(f'{DATASET}/Ventricle_L.nii.gz')
        ventricle_L_data = ventricle_L.get_fdata()
        
        ventricle_R = nib.load(f'{DATASET}/Ventricle_R.nii.gz')
        ventricle_R_data = ventricle_R.get_fdata()
        
        # 計算第三腦室
        third_ventricle = ventricles_data - ventricle_L_data - ventricle_R_data
        third_ventricle = keep_largest_island(third_ventricle, 100)
        third_ventricle_path = f'{DATASET}/Third_Ventricle.nii.gz'
        nib.save(nib.Nifti1Image(third_ventricle, ventricles.affine, ventricles.header), third_ventricle_path)
        result['output_files'].append(third_ventricle_path)
        
        # 計算第四腦室
        fourth_ventricle = ventricles_data - ventricle_L_data - ventricle_R_data - third_ventricle
        fourth_ventricle = keep_largest_island(fourth_ventricle, 100)
        fourth_ventricle_path = f'{DATASET}/Fourth_Ventricle.nii.gz'
        nib.save(nib.Nifti1Image(fourth_ventricle, ventricles.affine, ventricles.header), fourth_ventricle_path)
        result['output_files'].append(fourth_ventricle_path)
        
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        logger.info(f"完成處理病患: {patient_id} (耗時: {result['processing_time']:.2f}秒)")
        
    except Exception as e:
        result['success'] = False
        result['error_message'] = str(e)
        result['processing_time'] = time.time() - start_time
        logger.error(f"處理病患 {patient_id} 時發生錯誤: {e}")
    
    finally:
        # 清理記憶體
        try:
            if 'model' in locals():
                del model
            gc.collect()
        except:
            pass
    
    return result


def find_dicom_folders(parent_dir: str) -> List[str]:
    """
    在父目錄中尋找包含 DICOM 檔案的子資料夾
    
    Args:
        parent_dir (str): 父目錄路徑
        
    Returns:
        List[str]: DICOM 資料夾路徑列表
    """
    dicom_folders = []
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        logger.error(f"目錄不存在: {parent_dir}")
        return dicom_folders
    
    # 檢查是否包含 DICOM 檔案的目錄
    for item in parent_path.iterdir():
        if item.is_dir():
            # 檢查是否包含 .dcm 檔案或 DICOM 檔案
            has_dicom = False
            for file in item.iterdir():
                if file.is_file():
                    # 檢查副檔名
                    if file.suffix.lower() in ['.dcm', '.dicom'] or 'dicom' in file.name.lower():
                        has_dicom = True
                        break
                    # 檢查是否為沒有副檔名的 DICOM 檔案（使用 file 命令）
                    elif file.suffix == '':
                        try:
                            import subprocess
                            result = subprocess.run(['file', str(file)], capture_output=True, text=True, timeout=2)
                            if 'DICOM' in result.stdout:
                                has_dicom = True
                                break
                        except:
                            # 如果 file 命令失敗，檢查檔案名稱模式
                            if file.name.startswith('IMG') or file.name.startswith('IM'):
                                has_dicom = True
                                break
            
            if has_dicom:
                dicom_folders.append(str(item))
                logger.info(f"找到 DICOM 資料夾: {item}")
    
    return sorted(dicom_folders)


def batch_process_patients(input_paths: List[str], output_dir: Optional[str] = None, 
                         parallel: bool = False, max_workers: Optional[int] = None,
                         skip_existing: bool = False) -> List[Dict]:
    """
    批次處理多個病患的 DICOM 資料
    
    Args:
        input_paths (List[str]): DICOM 資料夾路徑列表
        output_dir (str, optional): 統一輸出目錄
        parallel (bool): 是否使用平行處理
        max_workers (int, optional): 最大工作執行緒數
        skip_existing (bool): 是否跳過已存在的結果
        
    Returns:
        List[Dict]: 所有病患的處理結果
    """
    logger.info(f"開始批次處理 {len(input_paths)} 個病患")
    results = []
    
    if parallel and len(input_paths) > 1:
        # 平行處理
        if max_workers is None:
            max_workers = min(len(input_paths), os.cpu_count() // 2)
        
        logger.info(f"使用平行處理，工作執行緒數: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_single_patient, path, output_dir, skip_existing): path 
                for path in input_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        logger.info(f"✓ 完成: {result['patient_id']}")
                    else:
                        logger.error(f"✗ 失敗: {result['patient_id']} - {result['error_message']}")
                except Exception as e:
                    logger.error(f"處理 {path} 時發生異常: {e}")
                    results.append({
                        'patient_id': Path(path).name,
                        'dataset_path': path,
                        'success': False,
                        'error_message': str(e),
                        'processing_time': 0,
                        'output_files': []
                    })
    else:
        # 序列處理
        for i, path in enumerate(input_paths, 1):
            logger.info(f"處理進度: {i}/{len(input_paths)}")
            result = process_single_patient(path, output_dir, skip_existing)
            results.append(result)
            
            if result['success']:
                logger.info(f"✓ 完成: {result['patient_id']}")
            else:
                logger.error(f"✗ 失敗: {result['patient_id']} - {result['error_message']}")
    
    return results


def generate_batch_report(results: List[Dict], output_dir: Optional[str] = None) -> str:
    """
    生成批次處理報告
    
    Args:
        results (List[Dict]): 處理結果列表
        output_dir (str, optional): 報告輸出目錄
        
    Returns:
        str: 報告檔案路徑
    """
    # 統計結果
    total_patients = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_patients - successful
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / total_patients if total_patients > 0 else 0
    
    # 生成報告
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""
腦部分割批次處理報告
生成時間: {report_time}

=== 處理摘要 ===
總病患數: {total_patients}
成功處理: {successful}
處理失敗: {failed}
成功率: {successful/total_patients*100:.1f}%
總處理時間: {total_time:.2f} 秒
平均處理時間: {avg_time:.2f} 秒/病患

=== 詳細結果 ===
"""
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        report_content += f"{status} {result['patient_id']}: {result['processing_time']:.2f}秒"
        if not result['success']:
            report_content += f" (錯誤: {result['error_message']})"
        report_content += "\n"
    
    if failed > 0:
        report_content += "\n=== 失敗案例 ===\n"
        for result in results:
            if not result['success']:
                report_content += f"{result['patient_id']}: {result['error_message']}\n"
    
    # 儲存報告
    if output_dir:
        report_path = Path(output_dir) / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    else:
        report_path = Path(f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"批次處理報告已儲存: {report_path}")
    return str(report_path)


def parse_arguments():
    """
    解析命令列參數
    """
    parser = argparse.ArgumentParser(
        description='腦部 CT 影像自動分割系統 - 支援單一或批次處理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  單一資料夾處理:
    python befor_processes.py --paths /path/to/dicom/folder
    
  多個資料夾處理:
    python befor_processes.py --paths /path/to/dicom1 /path/to/dicom2 /path/to/dicom3
    
  自動發現並批次處理:
    python befor_processes.py --batch /path/to/parent/folder
    
  從檔案讀取路徑列表:
    python befor_processes.py --input-list patients.txt
    
  平行處理:
    python befor_processes.py --parallel --max-workers 4 --paths /path/to/dicom1 /path/to/dicom2
        """
    )
    
    # 主要輸入參數
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--paths', nargs='+', metavar='PATH', help='DICOM 資料夾路徑（支援多個）')
    group.add_argument('--batch', metavar='DIR', help='自動發現父目錄中的 DICOM 資料夾')
    group.add_argument('--input-list', metavar='FILE', help='從檔案讀取 DICOM 資料夾路徑列表')
    
    # 輸出設定
    parser.add_argument('--output-dir', metavar='DIR', help='統一輸出目錄')
    parser.add_argument('--skip-existing', action='store_true', help='跳過已存在的結果檔案')
    
    # 平行處理設定
    parser.add_argument('--parallel', action='store_true', help='啟用平行處理')
    parser.add_argument('--max-workers', type=int, metavar='N', help='最大工作執行緒數（預設為 CPU 核心數的一半）')
    
    # 其他選項
    parser.add_argument('--generate-report', action='store_true', default=True, help='生成處理報告')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日誌等級')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # 設定日誌等級
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 收集輸入路徑
    input_paths = []
    
    if args.paths:
        # 直接指定的路徑
        input_paths = args.paths
    elif args.batch:
        # 自動發現模式
        input_paths = find_dicom_folders(args.batch)
        if not input_paths:
            logger.error(f"在 {args.batch} 中未找到 DICOM 資料夾")
            sys.exit(1)
    elif args.input_list:
        # 從檔案讀取
        try:
            with open(args.input_list, 'r', encoding='utf-8') as f:
                input_paths = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"找不到輸入檔案: {args.input_list}")
            sys.exit(1)
    
    # 驗證輸入路徑
    valid_paths = []
    for path in input_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            logger.warning(f"路徑不存在，跳過: {path}")
    
    if not valid_paths:
        logger.error("沒有有效的輸入路徑")
        sys.exit(1)
    
    input_paths = valid_paths
    
    # 向後相容性：如果只有一個路徑且沒有特殊參數，使用原始邏輯
    if len(input_paths) == 1 and not any([args.output_dir, args.parallel, args.skip_existing]):
        logger.info("使用單一病患處理模式（向後相容）")
        result = process_single_patient(input_paths[0])
        if result['success']:
            logger.info("處理完成")
            sys.exit(0)
        else:
            logger.error(f"處理失敗: {result['error_message']}")
            sys.exit(1)
    
    # 批次處理模式
    logger.info(f"批次處理模式，共 {len(input_paths)} 個病患")
    
    start_time = time.time()
    results = batch_process_patients(
        input_paths=input_paths,
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing
    )
    total_time = time.time() - start_time
    
    # 生成報告
    if args.generate_report:
        report_path = generate_batch_report(results, args.output_dir)
    
    # 輸出摘要
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"批次處理完成")
    logger.info(f"總耗時: {total_time:.2f} 秒")
    logger.info(f"成功: {successful}, 失敗: {failed}")
    logger.info(f"成功率: {successful/len(results)*100:.1f}%")
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

    