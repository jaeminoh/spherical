from esm import ESM2, Alphabet
import torch


class Scores:
    def __init__(self, model: ESM2, tokenizer: Alphabet):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_converter = self.tokenizer.get_batch_converter()

    def compute_ids(self, wt: str, mt: str):
        *_, toks = self.batch_converter([("wt", wt), ("mt", mt)])
        wt_ids, mt_ids = toks.to(next(self.model.parameters()).device)
        return wt_ids, mt_ids, (wt_ids != mt_ids)

    def compute_logits(self, input_ids: torch.Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids[None]  # (1, seq_len + 2)
        with torch.no_grad():
            logits = self.model(input_ids)["logits"]
        return logits.squeeze()

    def compute_score(self, log_probs, wt_ids, mt_ids):
        rows = torch.arange(log_probs.shape[0])
        assert len(rows) == len(wt_ids) == len(mt_ids), "Size mismatch!"
        log_probs_wt = log_probs[rows, wt_ids]
        log_probs_mt = log_probs[rows, mt_ids]
        return (log_probs_mt - log_probs_wt).sum()

    def get_input_ids(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, wt: str, mt: str):
        wt_ids, mt_ids, mutated = self.compute_ids(wt, mt)
        input_ids = self.get_input_ids(wt_ids, mt_ids, mutated)
        logits = self.compute_logits(input_ids)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) * mutated[:, None]
        return self.compute_score(log_probs, wt_ids, mt_ids)


class MaskedMarginal(Scores):
    r"""
    $$\sum_{i \in M} \log p(x_i^{mt} | x_{\setminus M}) - \log p(x_i^{wt} | x_{\setminus M})$$
    """

    def __init__(self, model: ESM2, tokenizer: Alphabet):
        super().__init__(model, tokenizer)

    def get_input_ids(self, wt_ids, mt_ids, mutated):
        return (~mutated) * wt_ids + mutated * self.tokenizer.mask_idx


class MutantMarginal(Scores):
    r"""
    \sum_{i \in M} \log p(x_i^{mt} | x^{mt}) - \log p(x_i^{wt} | x^{mt})
    """

    def __init__(self, model: ESM2, tokenizer: Alphabet):
        super().__init__(model, tokenizer)

    def get_input_ids(self, wt_ids, mt_ids, mutated):
        return mt_ids


class WildtypeMarginal(Scores):
    r"""
    $$\sum_{i \in M} \log p(x_i^{mt} | x^{wt}) - \log p(x_i^{wt} | x^{wt})$$
    """

    def __init__(self, model: ESM2, tokenizer: Alphabet):
        super().__init__(model, tokenizer)

    def get_input_ids(self, wt_ids, mt_ids, mutated):
        return wt_ids


class Entropy(Scores):
    r"""
    For a position i,
    $$\sum_{a \in A} - p(a) \log p(a)$$
    """

    def __init__(self, model: ESM2, tokenizer: Alphabet):
        super().__init__(model, tokenizer)

    def __call__(self, sequence: str):
        *_, input_ids = self.batch_converter([("seq", sequence)])
        input_ids = input_ids.to(next(self.model.parameters()).device)
        logits = self.compute_logits(input_ids).squeeze()  # (seq_len + 2, vocab_size)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        entropy = -(probs * log_probs).sum(dim=-1)  # (seq_len + 2,)
        return entropy[1:-1]  # (seq_len,) exclude BOS and EOS


class ContactPredictor:
    def __init__(self, model: ESM2, alphabet: Alphabet):
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def get_contact_matrix(self, name: str, sequence: str):
        *_, batch_tokens = self.batch_converter([(name, sequence)])
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        with torch.no_grad():
            predictions = self.model.predict_contacts(batch_tokens)[0]
        return predictions
