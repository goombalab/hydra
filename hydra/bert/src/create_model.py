import os 
import sys
# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bert_layers as bert_layers_module
import configuration_bert as configuration_bert_module

def create_model(model_config):
    pretrained_model_name = 'bert-base-uncased'
    config = configuration_bert_module.BertConfig.from_pretrained(
        pretrained_model_name, **model_config)
    for key, value in model_config.items():
        config.update({f'{key}': value})
    # https://twitter.com/karpathy/status/1621578354024677377?lang=en
    # highest token is 
    # 5447
    # because of karpathy, we use 5504
    config.vocab_size = 5504
    model = bert_layers_module.BertForMaskedLM(config)
    return model
