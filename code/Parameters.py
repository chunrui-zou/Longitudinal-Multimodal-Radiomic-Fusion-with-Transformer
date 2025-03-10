import torch
import numpy as np

	
ML={
	"ModalMerge":{
		"modality":{
			"cr": {"d_feature": 261,"hidden_sizes": [64,32]},
			"ct":  {"d_feature": 271,"hidden_sizes": [64,32]}
		},
		"n_heads":3,
		"n_enc_lays":3,
		"n_heads_delta" :3,
		"n_enc_lays_delta":3, # last encoder_delta to convert to a vector
		"d_model":32,
		"padding":-1,
		"max_num_mods":100,
		"d_delta": 32, # should not be too different from d_model, it is used to calc Q
		"padding": -1,
		"dropout": 0.0,
		"d_k":None,
		"d_v":None,
		"d_ff":None,
		"dropout": 0.0,
		"max_seq_len":200

	},
	"LongitudinalMerge":{
		"d_model":32,
		"padding": -1,
		"n_heads":3,
		"n_enc_lays":3,
		"d_delta":32, # maybe 16
		"d_k":None,
		"d_v":None,
		"d_ff":None,
		"dropout": 0.0,
		"max_seq_len":200
	},
	"Task":{
			"num_tasks":3,
    		"n_heads":3,
    		"d_model":32,
    		"output_size":1,
    		"n_enc_lays":1,
    		"d_delta": 4, # control the cov attention, 2 for next trying
    		"d_k": None,
    		"d_v": None,
    		"d_ff": None,
    		"hidden_sizes": None, # the defaults are not bad
    		"dropout": 0.2
            
	}
	
}






