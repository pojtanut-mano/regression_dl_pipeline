
config = {
    # Config for main
    'seed': 1234,

    # Config for Dataset
    'from_csv': False,
    'target': ['price'],

    # If loading from csv
    'source_filename': 'data.csv',
    'return_X_y': True,

    # Config for Preprocessing
    # Transformation of X
    'fillNaN': True,

    'PCA': False,
    'PCA_threshold': 0.95,

    'normalization': True,

    'remove_corr_cols': True,
    'corr_threshold': True,

    # Transformation of y
    'BoxCox': False,

    # Train test config
    'test_size': 0.2,
    'shuffle': True

    # Config for NetworkBuilder


    # Config for PostProcessing

}