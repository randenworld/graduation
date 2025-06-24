from Data_Preprocessing import keep_largest_island,run_total
from train import predict_2_5d_single_patient,build_2_5d_unet_model
import sys
import nibabel as nib
#預測 左右腦室 全腦室  
#從全腦室中去除左右腦室 利用keep_largest_island 取出三腦室
#再去除三腦室 取出四腦室

if __name__ == "__main__":
    #dicom  資料夾路徑
    DATASET = sys.argv[1]
    predict_file = run_total(DATASET)
    # predict_file = sys.argv[2]
    model = build_2_5d_unet_model((512,512,1))
    file_path = DATASET+"/original.nii.gz"
    MODEL_PATH = "train_2D/predict_argv/CSF.keras"
    model.load_weights(MODEL_PATH)
    predict_2_5d_single_patient(model,file_path ,False,name='CSF')

    MODEL_PATH = "train_2D/predict_argv/Ventricles.keras"
    model.load_weights(MODEL_PATH)
    predict_2_5d_single_patient(model, file_path,False,name='Ventricles')

    # MODEL_PATH = ""
    # model.load_weights(MODEL_PATH)
    # predict_2_5d_single_patient(model, file_path,False,name='Falx')

    MODEL_PATH = "train_2D/predict_argv/Ventricle_L.keras"
    model.load_weights(MODEL_PATH)
    predict_2_5d_single_patient(model, file_path,False,name='Ventricle_L')

    MODEL_PATH = "train_2D/predict_argv/Ventricle_R.keras"
    model.load_weights(MODEL_PATH)
    predict_2_5d_single_patient(model, file_path,False,name='Ventricle_R')

    ventricles = nib.load(f'{DATASET}/Ventricles.nii.gz')
    ventricles_data = ventricles.get_fdata()

    ventricle_L = nib.load(f'{DATASET}/Ventricle_L.nii.gz')
    ventricle_L_data = ventricle_L.get_fdata()

    ventricle_R = nib.load(f'{DATASET}/Ventricle_R.nii.gz')
    ventricle_R_data = ventricle_R.get_fdata()

    third_ventricle = ventricles_data - ventricle_L_data - ventricle_R_data
    third_ventricle = keep_largest_island(third_ventricle,100)
    nib.save(nib.Nifti1Image(third_ventricle, ventricles.affine, ventricles.header), DATASET+'/Third_Ventricle.nii.gz')

    fourth_ventricle = ventricles_data - ventricle_L_data - ventricle_R_data - third_ventricle
    fourth_ventricle = keep_largest_island(fourth_ventricle,100)

    nib.save(nib.Nifti1Image(fourth_ventricle, ventricles.affine, ventricles.header), DATASET+'/Fourth_Ventricle.nii.gz')

    