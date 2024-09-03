import torch 
import numpy as np
from collections.abc import Mapping

#from transformers import  DataCollatorForLanguageModeling
#data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability = 0.15)
torch.manual_seed(0)

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)


    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class DataCollatorForLanguageModelingSpan():
    
    def __init__(self, tokenizer, mlm, mlm_probability, span_length):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.span_length = span_length
        self.mlm_probability= mlm_probability
        self.pad_to_multiple_of = span_length

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        batch["species_id"] = batch["species_id"].reshape(-1, 1)
        
        #we have to clone after expand, because otherwise some in-place operations don't work
        batch["species_id"] = batch["species_id"].expand(batch["input_ids"].shape).clone()
#        pad_length = batch["input_ids"].shape[1] - batch["species_id"].shape[1]
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        import torch
        
        original_inputs = inputs.clone()
        labels = inputs.clone()
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 10% of the time, we leave the input unchanged
        probability_matrix = torch.full(labels.shape, self.mlm_probability*0.1)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool().numpy()
        # to ensure that we create spans, we convolve with a filter length of 6
        # we convert back to bool to account for overlaps (anything bigger than 0 gets masked)
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * self.span_length, mode = 'same' ),axis = 1, arr = masked_indices).astype(bool) 
        masked_indices = torch.from_numpy(masked_indices)
        m_save = masked_indices.clone()
        
        # 10% of the time, we replace masked input tokens with random nt
        # create a random offset matrix (randint(1,4))
        offsets = torch.randint(1, 4, labels.shape)
        # multiply with a masking matrix to get masked offsets
        probability_matrix = torch.full(labels.shape, self.mlm_probability*0.1)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(m_save, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_offsets = masked_indices*offsets
        # make the modulo matrix which identifies the nucleotide
        modulo_matrix = torch.remainder(inputs - 5, 4) # adjust for special tokens
        # Get the masked shifts matrix
        masked_shifts = torch.remainder(modulo_matrix + masked_offsets, 4) - modulo_matrix
        # we now propagate the change to the next few tokens
        for i in range(self.span_length):
            shifted_shift = masked_shifts[:,:(masked_shifts.shape[1] - i)]
            inputs[:,i:] = (shifted_shift*(4**i)) + inputs[:,i:]
            masked_indices[:,i:] = shifted_shift + masked_indices[:,i:]
        masked_indices = masked_indices.bool()
        m_save = m_save + masked_indices
        
        # 80% of the time, we mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability*0.8) 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(m_save, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool().numpy()
        # to ensure that we create spans, we convolve with a filter length of 6
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * self.span_length, mode = 'same' ),axis = 1, arr = masked_indices).astype(bool) 
        masked_indices = torch.from_numpy(masked_indices)
        
        # aggregate all the positions where we want loss
        m_final = masked_indices + m_save 
        labels[~m_final] = -100  # We only compute loss on masked tokens
        # we actually replace with the mask token
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # restore the special tokens
        inputs[special_tokens_mask] = original_inputs[special_tokens_mask]
        labels[special_tokens_mask] = -100
        
        return inputs, labels

#
# For debugging the species embedding code

if __name__ == "__main__":
    from datasets import load_from_disk
    from collate import DataCollatorForLanguageModelingSpan
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    dataset = load_from_disk("bert/batch_embed")
    dataset = dataset.remove_columns(["species_name", "__index_level_0__"])

    tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")
    data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.02, span_length = 6)

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

    sample = next(iter(dataloader))
    
    print(sample['input_ids'][-1])
    print(sample['species_id'][-1])