from datasets import load_from_disk, load_dataset
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer
from text_data import ConcatenatedSequenceCollatorWrapper
from collate import DataCollatorForLanguageModelingSpan

from torch.utils.data import DataLoader
from omegaconf import DictConfig
from composer.utils import dist


def build_dna_dataloader(
    cfg: DictConfig,
    device_batch_size: int,
):
    assert cfg.name == 'dna', f'Tried to build dna dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset.py')

    #return test_set(device_batch_size)
    assert cfg.dataset.local is not None, "No local dataset provided"
    dataset = load_from_disk(cfg.dataset.local)
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset[cfg.dataset.split]

    dataset = dataset.remove_columns(["species_name", "__index_level_0__"])

    tokenizer = transformers.AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")
    mlm_probability = cfg.dataset.get('mlm_probability', None)
    collate_fn = DataCollatorForLanguageModelingSpan(
        tokenizer=tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability,
        span_length=6)
    print("==Using the correct data collator")
    

    sampler = dist.get_sampler(dataset, shuffle=cfg.dataset.shuffle, drop_last=cfg.drop_last)

    eos_token_id = cfg.dataset.get('eos_token_id')
    bos_token_id = cfg.dataset.get('bos_token_id')
    if (eos_token_id is not None) or (bos_token_id is not None):
        raise Exception("wrong collator")
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
        sampler=sampler
    )

def test_set(batch_size) : 
    eli5 = load_dataset("eli5_category")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    eli5 = eli5.flatten()
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])
    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched = True,
        num_proc = 4,
        remove_columns = eli5["train"].column_names
    )

    # concatenate sequences
    block_size = 128


    def group_texts(examples):

        # Concatenate all texts.

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can

        # customize this part to your needs.

        if total_length >= block_size:

            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size.

        result = {

            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]

            for k, t in concatenated_examples.items()

        }

        return result
    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    collate_fn = DataCollatorForLanguageModeling(
    tokenizer=tokenizer)
    loader = DataLoader(lm_dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    return loader