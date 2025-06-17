import nibabel as nib
import numpy as np
import pandas as pd
import os
import re
import glob

def identify_region_and_patient(filename):
    """
    從檔案名稱識別解剖區域和病人編號
    
    參數:
    filename: 檔案名稱
    
    返回:
    region_name: 中文解剖區域名稱
    patient_id: 病人編號
    """
    
    # 解剖區域對應表
    region_mapping = {
        'CSF': '腦脊髓液',
        'Falx': '大腦鐮',
        'Fourth-ventricle': '第四腦室',
        'Tentorium': '小腦天幕',
        'Third-ventricle': '第三腦室',
        'Ventricle_L': '左腦室',
        'Ventricle_R': '右腦室',
        'Basal-cistern': '基底池',
        'Ventricles': '腦室'
    }
    
    # 提取病人編號（檔案名稱中的數字）
    numbers = re.findall(r'_(\d+)\.', filename)
    patient_id = numbers[0] if numbers else "未知"
    
    # 識別解剖區域
    region_name = "未知區域"
    
    # 移除 "mask_" 前綴後進行比對
    filename_clean = filename.replace('mask_', '')
    
    for key, value in region_mapping.items():
        if key.lower() in filename_clean.lower():
            region_name = value
            break
    
    return region_name, patient_id

def calculate_volume_from_nii(nii_file_path):
    """
    計算NII檔案的3D體積
    
    參數:
    nii_file_path: NII檔案的完整路徑
    
    返回:
    volume_mm3: 體積（立方毫米）
    volume_ml: 體積（毫升）
    voxel_count: 體素數量
    """
    
    try:
        # 載入NII檔案
        nii_img = nib.load(nii_file_path)
        data = nii_img.get_fdata()
        header = nii_img.header
        
        # 獲取體素尺寸
        voxel_dims = header.get_zooms()
        voxel_volume = np.prod(voxel_dims)  # 單個體素的體積（mm³）
        
        # 計算非零體素的數量
        if np.all(np.isin(data, [0, 1])):
            # 二值化情況
            structure_voxels = np.sum(data == 1)
        else:
            # 計算所有非零體素
            structure_voxels = np.sum(data > 0)
        
        # 計算總體積
        volume_mm3 = structure_voxels * voxel_volume
        volume_ml = volume_mm3 / 1000  # 轉換為毫升
        
        return volume_mm3, volume_ml, structure_voxels
        
    except Exception as e:
        print(f"處理檔案 {nii_file_path} 時發生錯誤: {e}")
        return 0, 0, 0

def calculate_ventricle_symmetry(df):
    """
    計算左右腦室的對稱性分析
    
    參數:
    df: 包含體積分析結果的DataFrame
    
    返回:
    symmetry_results: 包含對稱性分析的DataFrame
    """
    
    # 找出有左右腦室數據的病人
    left_ventricle_data = df[df['解剖區域'] == '左腦室'].copy()
    right_ventricle_data = df[df['解剖區域'] == '右腦室'].copy()
    
    # 合併左右腦室數據
    merged_data = pd.merge(
        left_ventricle_data[['病人編號', '體積 (ml)']].rename(columns={'體積 (ml)': '左腦室體積 (ml)'}),
        right_ventricle_data[['病人編號', '體積 (ml)']].rename(columns={'體積 (ml)': '右腦室體積 (ml)'}),
        on='病人編號',
        how='inner'
    )
    
    if merged_data.empty:
        print("警告: 沒有找到配對的左右腦室數據")
        return pd.DataFrame()
    
    # 計算對稱性指標
    symmetry_results = []
    
    for _, row in merged_data.iterrows():
        left_vol = row['左腦室體積 (ml)']
        right_vol = row['右腦室體積 (ml)']
        
        # 計算基本指標
        total_vol = left_vol + right_vol
        ratio_l_to_r = left_vol / right_vol if right_vol > 0 else float('inf')
        ratio_r_to_l = right_vol / left_vol if left_vol > 0 else float('inf')
        
        # 計算對稱性指數 (Asymmetry Index)
        # AI = |L - R| / (L + R) * 100%
        asymmetry_index = abs(left_vol - right_vol) / total_vol * 100 if total_vol > 0 else 0
        
        # 判斷對稱性等級
        if asymmetry_index <= 5:
            symmetry_grade = "高度對稱"
        elif asymmetry_index <= 10:
            symmetry_grade = "中度對稱"
        elif asymmetry_index <= 20:
            symmetry_grade = "輕度不對稱"
        else:
            symmetry_grade = "明顯不對稱"
        
        # 判斷哪側較大
        if left_vol > right_vol:
            dominant_side = "左側較大"
            volume_difference = left_vol - right_vol
        elif right_vol > left_vol:
            dominant_side = "右側較大"
            volume_difference = right_vol - left_vol
        else:
            dominant_side = "完全對稱"
            volume_difference = 0
        
        symmetry_results.append({
            '病人編號': row['病人編號'],
            '左腦室體積 (ml)': round(left_vol, 4),
            '右腦室體積 (ml)': round(right_vol, 4),
            '總腦室體積 (ml)': round(total_vol, 4),
            '左/右比值': round(ratio_l_to_r, 3),
            '右/左比值': round(ratio_r_to_l, 3),
            '體積差異 (ml)': round(volume_difference, 4),
            '不對稱指數 (%)': round(asymmetry_index, 2),
            '對稱性等級': symmetry_grade,
            '優勢側': dominant_side
        })
    
    return pd.DataFrame(symmetry_results)

def batch_analyze_brain_volumes(directory_path="."):
    """
    批次分析指定目錄下的所有NII.GZ檔案
    
    參數:
    directory_path: 包含NII檔案的目錄路徑
    
    返回:
    DataFrame: 包含所有分析結果的表格
    """
    
    # 搜尋所有.nii.gz檔案
    nii_files = glob.glob(os.path.join(directory_path, "*.nii.gz"))
    
    if not nii_files:
        print(f"在目錄 {directory_path} 中找不到 .nii.gz 檔案")
        return pd.DataFrame()
    
    results = []
    
    print("開始批次分析...")
    print("="*60)
    
    for i, file_path in enumerate(nii_files, 1):
        filename = os.path.basename(file_path)
        print(f"處理第 {i}/{len(nii_files)} 個檔案: {filename}")
        
        # 識別區域和病人編號
        region_name, patient_id = identify_region_and_patient(filename)
        
        # 計算體積
        volume_mm3, volume_ml, voxel_count = calculate_volume_from_nii(file_path)
        
        # 儲存結果
        results.append({
            '病人編號': patient_id,
            '解剖區域': region_name,
            '體素數量': voxel_count,
            '體積 (mm³)': round(volume_mm3, 2),
            '體積 (ml)': round(volume_ml, 4),
            '體積 (μl)': round(volume_ml * 1000, 2)
        })
        
    # 建立DataFrame
    df = pd.DataFrame(results)
    
    # 按病人編號和解剖區域排序
    df['病人編號_數字'] = pd.to_numeric(df['病人編號'], errors='coerce')
    df = df.sort_values(['病人編號_數字', '解剖區域']).drop('病人編號_數字', axis=1)
    df = df.reset_index(drop=True)
    
    return df

def generate_summary_report(df):
    """
    生成摘要報告
    """
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("分析摘要報告")
    print("="*60)
    
    # 總體統計
    print(f"總共分析檔案數: {len(df)}")
    print(f"涉及病人數: {df['病人編號'].nunique()}")
    print(f"涉及解剖區域: {df['解剖區域'].nunique()}")
    
    # 各區域統計
    print("\n各解剖區域統計:")
    
    for region in df['解剖區域'].unique():
        region_data = df[df['解剖區域'] == region]['體積 (ml)']
        print(f"\n{region}:")
        print(f"  樣本數: {len(region_data)}")
        print(f"  平均體積: {region_data.mean():.4f} ml")
        if len(region_data) > 1:
            print(f"  標準差: {region_data.std():.4f} ml")
        print(f"  體積範圍: {region_data.min():.4f} - {region_data.max():.4f} ml")

def generate_symmetry_report(symmetry_df):
    """
    生成腦室對稱性報告
    """
    if symmetry_df.empty:
        print("\n無法生成腦室對稱性報告 - 沒有配對的左右腦室數據")
        return
    
    print("\n" + "="*60)
    print("腦室對稱性分析報告")
    print("="*60)
    
    # 對稱性等級統計
    symmetry_counts = symmetry_df['對稱性等級'].value_counts()
    print(f"分析的病人數: {len(symmetry_df)}")
    print("\n對稱性等級分布:")
    for grade, count in symmetry_counts.items():
        percentage = (count / len(symmetry_df)) * 100
        print(f"  {grade}: {count} 人 ({percentage:.1f}%)")
    
    # 優勢側統計
    print(f"\n優勢側分布:")
    dominance_counts = symmetry_df['優勢側'].value_counts()
    for side, count in dominance_counts.items():
        percentage = (count / len(symmetry_df)) * 100
        print(f"  {side}: {count} 人 ({percentage:.1f}%)")
    
    # 數值統計
    print(f"\n不對稱指數統計:")
    print(f"  平均: {symmetry_df['不對稱指數 (%)'].mean():.2f}%")
    print(f"  標準差: {symmetry_df['不對稱指數 (%)'].std():.2f}%")
    print(f"  範圍: {symmetry_df['不對稱指數 (%)'].min():.2f}% - {symmetry_df['不對稱指數 (%)'].max():.2f}%")
    
    print(f"\n體積比值統計:")
    # 計算較小比值（總是 <= 1）
    smaller_ratios = np.minimum(symmetry_df['左/右比值'], symmetry_df['右/左比值'])
    print(f"  平均較小側/較大側比值: {smaller_ratios.mean():.3f}")
    print(f"  標準差: {smaller_ratios.std():.3f}")

def save_results(df, symmetry_df=None):
    """
    儲存結果到CSV檔案
    """
    output_folder = "volume_analysis"
    
    # 創建輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 儲存主要分析結果
    if not df.empty:
        main_output = os.path.join(output_folder, "brain_volume_analysis.csv")
        df.to_csv(main_output, index=False, encoding='utf-8-sig')
        print(f"\n主要分析結果已儲存至: {main_output}")
    
    # 儲存對稱性分析結果
    if symmetry_df is not None and not symmetry_df.empty:
        symmetry_output = os.path.join(output_folder, "ventricle_symmetry_analysis.csv")
        symmetry_df.to_csv(symmetry_output, index=False, encoding='utf-8-sig')
        print(f"腦室對稱性分析結果已儲存至: {symmetry_output}")
    
    if df.empty and (symmetry_df is None or symmetry_df.empty):
        print("\n沒有資料可以儲存")

# 主程式
if __name__ == "__main__":
    # 設定檔案目錄路徑
    directory_path = "volume"  # 當前目錄，可以修改為你的檔案所在目錄
    
    print("腦部結構體積批次分析工具 (含腦室對稱性分析)")
    print("="*60)
    
    # 執行批次分析
    results_df = batch_analyze_brain_volumes(directory_path)
    
    if not results_df.empty:
        # 顯示結果表格
        print("\n詳細分析結果:")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # 計算腦室對稱性
        symmetry_df = calculate_ventricle_symmetry(results_df)
        
        # 生成摘要報告
        generate_summary_report(results_df)
        
        # 生成對稱性報告
        if not symmetry_df.empty:
            generate_symmetry_report(symmetry_df)
            
            # 顯示對稱性詳細結果
            print("\n" + "="*60)
            print("腦室對稱性詳細分析:")
            print("="*60)
            print(symmetry_df.to_string(index=False))
        
        # 儲存結果
        save_results(results_df, symmetry_df)
        
        # 顯示各病人的數據
        print("\n" + "="*60)
        print("按病人分組的結果:")
        print("="*60)
        for patient_id in sorted(results_df['病人編號'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            patient_data = results_df[results_df['病人編號'] == patient_id]
            print(f"\n病人 {patient_id}:")
            for _, row in patient_data.iterrows():
                print(f"  {row['解剖區域']}: {row['體積 (ml)']} ml ({row['體積 (μl)']} μl)")
            
            # 如果有對稱性數據，也顯示
            if not symmetry_df.empty:
                patient_symmetry = symmetry_df[symmetry_df['病人編號'] == patient_id]
                if not patient_symmetry.empty:
                    sym_data = patient_symmetry.iloc[0]
                    print(f"  腦室對稱性: {sym_data['對稱性等級']} (不對稱指數: {sym_data['不對稱指數 (%)']}%)")
                    print(f"  優勢側: {sym_data['優勢側']}")
    
    else:
        print("沒有找到可分析的檔案")

# 單獨測試特定檔案
def test_single_file(filename):
    """
    測試單一檔案的分析功能
    """
    region_name, patient_id = identify_region_and_patient(filename)
    print(f"檔案: {filename}")
    print(f"識別的區域: {region_name}")
    print(f"病人編號: {patient_id}")
    print("-" * 40)

# 測試檔案名稱識別