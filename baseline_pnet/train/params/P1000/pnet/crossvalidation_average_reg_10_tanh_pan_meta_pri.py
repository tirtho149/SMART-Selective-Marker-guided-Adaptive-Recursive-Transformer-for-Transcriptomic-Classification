from model.builders.prostate_models import build_pnet2

task = 'classification_binary'

import os as _os
_DATA_ROOT = _os.environ.get('BIOMOR_DATA_ROOT',
                             '/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/data')

data_base = {'id': 'PAN_META_PRI', 'type': 'brca',
             'params': {
                 'data_dir': _os.path.join(_DATA_ROOT, 'pan_meta_pri'),
                 'labels_filename': 'patient_labels.csv',
                 'selected_genes_filename': None,
                 'val_size': 0.10,
                 'test_size': 0.2,
                 'random_state': 42,
                 'zscore_cnv': False,
             }
             }
data = [data_base]

n_hidden_layers = 1
base_dropout = 0.2
wregs = [0.001] * 7
loss_weights = [2, 7]
wreg_outcomes = [0.01] * 6
pre = {'type': None}

nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            'build_fn': build_pnet2,
            'model_params': {
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False
            },
            'fitting_params': dict(samples_per_epoch=10,
                                   select_best_model=False,
                                   monitor='val_o2_f1',
                                   verbose=2,
                                   epoch=200,
                                   shuffle=True,
                                   batch_size=16,
                                   save_name='pnet',
                                   debug=False,
                                   save_gradient=False,
                                   class_weight='auto',
                                   n_outputs=n_hidden_layers + 1,
                                   prediction_output='average',
                                   early_stop=False,
                                   reduce_lr=False,
                                   reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                   lr=1e-4,
                                   max_f1=True
                                   ),
            'feature_importance': None
        },
}
features = {}
models = [nn_pathway]

pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}
