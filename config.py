data_dict = {
    0: '100Leaves',
}


def get_config(flag=0):
    """Determine the parameter information of the model"""
    data_name = data_dict[flag]
    if data_name in ['100Leaves']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.5,
            n_clustering=100,
            view_num=3,
            method='heat',
            o_dim=32,
            training=dict(
                lr=1.0e-4,
                epoch=200,
                communication = 3,
                batch_size = 128,
                lamb_re = 1.0,
                lamb_re_f = 0.1,
                lamb_kl = 1.0,
                data_seed=3,
            ),
            Autoencoder=dict(
                gatEncoder1=[
                    [64, 1024, 1024, 1024, 32],
                    [64, 1024, 1024, 1024, 32],
                    [64, 1024, 1024, 1024, 32],
                ],
                gatEncoder2=[
                    [64, 1024, 1024, 1024, 32],
                    [64, 1024, 1024, 1024, 32],
                    [64, 1024, 1024, 1024, 32],
                ],
                activations=[
                    'relu',
                    'relu',
                    'relu',
                ],
                batchnorm=True,
            )
        )
