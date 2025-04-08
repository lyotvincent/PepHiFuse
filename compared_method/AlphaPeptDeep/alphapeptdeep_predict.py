from peptdeep.model.featurize import (
    get_batch_aa_indices,
    get_batch_mod_feature
)

from peptdeep.settings import model_const

import peptdeep.model.model_interface as model_base
import peptdeep.model.building_block as building_block

mod_feature_size = len(model_const['mod_elements'])

import torch
import pandas as pd

from peptdeep.model.generic_property_prediction import (
    ModelInterface_for_Generic_AASeq_BinaryClassification,
    ModelInterface_for_Generic_AASeq_Regression,
    ModelInterface_for_Generic_ModAASeq_BinaryClassification,
    ModelInterface_for_Generic_ModAASeq_Regression,
)
from peptdeep.model.generic_property_prediction import (
    Model_for_Generic_AASeq_BinaryClassification_LSTM,
    Model_for_Generic_AASeq_BinaryClassification_Transformer,
    Model_for_Generic_AASeq_Regression_LSTM,
    Model_for_Generic_AASeq_Regression_Transformer,
    Model_for_Generic_ModAASeq_BinaryClassification_LSTM,
    Model_for_Generic_ModAASeq_BinaryClassification_Transformer,
    Model_for_Generic_ModAASeq_Regression_LSTM,
    Model_for_Generic_ModAASeq_Regression_Transformer,
)


class RT_LSTM_Module(torch.nn.Module):
    def __init__(self,
        dropout=0.2
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)

        hidden = 128
        self.rt_encoder = building_block.Encoder_26AA_Mod_CNN_LSTM_AttnSum(
            hidden
        )

        self.rt_decoder = building_block.Decoder_Linear(
            hidden,
            1
        )

    def forward(self,
        aa_indices,
        mod_x,
    ):
        x = self.rt_encoder(aa_indices, mod_x)
        x = self.dropout(x)

        return self.rt_decoder(x).squeeze(1)


class RT_Transformer_Module(torch.nn.Module):
    def __init__(self,
        dropout=0.2
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)

        hidden = 128
        self.encoder = building_block.Encoder_AA_Mod_Transformer_AttnSum(
            hidden
        )

        self.decoder = building_block.Decoder_Linear(
            hidden,1
        )

    def forward(self,
        aa_indices,
        mod_x,
    ):
        x = self.encoder(aa_indices, mod_x)
        x = self.dropout(x)

        return self.decoder(x).squeeze(1)


class RT_ModelInterface(model_base.ModelInterface):
    def __init__(self,
        model_class:torch.nn.Module=RT_LSTM_Module,
        dropout=0.1,
    ):
        super().__init__()
        self.build(
            model_class,
            dropout=dropout,
        )
        self.loss_func = torch.nn.L1Loss()
        self.target_column_to_train = 'rt_norm'
        self.target_column_to_predict = 'rt_pred'

    def _get_features_from_batch_df(self,
        batch_df: pd.DataFrame,
    ):
        aa_indices = torch.LongTensor(
            get_batch_aa_indices(
                batch_df['sequence'].values.astype('U')
            )
        )
        mod_x = torch.Tensor(
            get_batch_mod_feature(
                batch_df
            )
        )

        return aa_indices.to("cuda"), mod_x.to("cuda")

    def _get_targets_from_batch_df(self,
        batch_df: pd.DataFrame,
    ) -> torch.Tensor:
        return torch.Tensor(batch_df['rt_norm'].values).to("cuda")


irt_pep = pd.read_csv("SAL00141/SAL00141_test.tsv", sep='\t')
irt_pep = irt_pep.rename(columns={'x':'sequence', 'y':'rt'})

irt_pep['mods'] = ''
irt_pep['mod_sites'] = ''

irt_pep = irt_pep[~irt_pep.sequence.str.contains('1')]

rt_model = ModelInterface_for_Generic_AASeq_Regression(
    model_class=Model_for_Generic_AASeq_Regression_Transformer
)

rt_model.load("SAL00141_models/model.pth")

rt_model.target_column_to_predict = 'predicted_rt'

rt_model.predict(irt_pep)

irt_pep.to_csv("SAL00141/SAL00141_predicted.tsv", sep='\t', index=False)