class hparams:

    train_or_test = 'train'
    output_dir = 'logs/3d_seg'
    aug = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 100
    epochs_per_checkpoint = 10
    batch_size = 16
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 512,512,32 # if 2D: 256,256,1
    patch_size = 32,32,32 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    # fold_arch = '*.mhd'
    fold_arch = '*.nii.gz'


    save_arch = '.nii.gz'

    source_train_dir = 'E:\\xgh\\nnunet\\nnunet\\nnUNet\\DATSET\\nnUNet_raw\\Dataset002_Heart\\imagesTr'
    label_train_dir = 'E:\\xgh\\nnunet\\nnunet\\nnUNet\\DATSET\\nnUNet_raw\\Dataset002_Heart\\labelsTr'
    source_test_dir = 'E:\\xgh\\nnunet\\nnunet\\nnUNet\\DATSET\\nnUNet_raw\\Dataset002_Heart\\imagesTs'
    label_test_dir = 'E:\\xgh\\nnunet\\nnunet\\nnUNet\\DATSET\\nnUNet_raw\\Dataset002_Heart\\imagesTs_predlowres'


    output_dir_test = 'results/3d_seg'