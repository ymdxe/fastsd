import torch
from .util import norm_logits, sample


class KVCacheModel():
    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
        generator: torch.Generator | None = None,
        exact_top_k: bool = False,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._generator = generator
        self._exact_top_k = bool(exact_top_k)

    def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            # print("hello3")
            outputs = self._model(input_ids)
            # Keep normalized probabilities in FP32.  Model logits may be
            # BF16/FP16, but writing the distribution back into that storage
            # would change both the uploaded q block and the residual math.
            self._prob_history = outputs.logits[:, :, :self.vocab_size].float()
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                    exact_top_k=self._exact_top_k,
                )
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # print("hello4")
            # return the last token's logits
            cached_len = self._past_key_values.get_seq_length()
                
            last_input_id = input_ids[:, cached_len:]

            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits[:, :, :self.vocab_size].float()
            
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                    exact_top_k=self._exact_top_k,
                )
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(
        self,
        prefix: torch.Tensor,
        gamma: int,
        return_prob_rows: bool = False,
        sample_next: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        sampled_rows = []

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            if return_prob_rows:
                sampled_rows.append(q.detach().float())
            if sample_next:
                next_tok = sample(q, generator=self._generator)
            else:
                # Target verification only needs the forward probabilities;
                # its correction token is sampled by the shared rejection
                # core.  Keep the legacy return shape without consuming a
                # hidden global RNG draw.
                next_tok = torch.argmax(q, dim=-1, keepdim=True)
            x = torch.cat((x, next_tok), dim=1)
        if return_prob_rows:
            rows = (
                torch.cat(sampled_rows, dim=0)
                if sampled_rows
                else torch.empty((0, self.vocab_size), device=x.device, dtype=torch.float32)
            )
            return x, rows
        return x

    def reset(self):
        self._past_key_values = None
        self._prob_history = None

    @torch.no_grad()
    def generate(
        self, input: torch.Tensor, gamma: int, sample_next: bool = True
    ) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, sample_next=sample_next)
        return output

    @torch.no_grad()
    def generate_with_probs(
        self, input: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens and return the exact q row used for each one."""

        output, rows = self._generate_with_kvcache(input, gamma, return_prob_rows=True)
        return output, rows
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        self._past_key_values.crop(end_pos)
        self._prob_history = self._prob_history[:, :end_pos, :]
