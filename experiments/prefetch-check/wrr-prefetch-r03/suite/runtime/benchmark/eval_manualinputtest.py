import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import time
import random
from src.util import seed_everything, parse_arguments
from src.engine import Decoding

class ManualPromptEval(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        self.load_model()

    def load_data(self):
        pass  # not used

    def preprocess(self, input_text):
        return input_text.strip()

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:]
        else:
            generation = output_text[len(input_text):]

        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                generation = generation[:generation.index(stop_word)].strip()

        return input_text + "\n    " + generation.replace("\t", "    ")

    @torch.no_grad()
    def eval(self):
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        elif self.args.eval_mode == "para_sd_wo_1":
            decoding = self.parallel_speculative_decoding_without_strategy_1
        elif self.args.eval_mode == "para_sd_wo_2":
            decoding = self.parallel_speculative_decoding_without_strategy_2
        elif self.args.eval_mode == "rc_para_sd":
            decoding = self.parallel_speculative_decoding_RC
        else:
            raise NotImplementedError

        while True:
            input_text = input("Please input your prompt:\n> ")
            if input_text.strip().lower() in ["exit", "quit"]:
                break

            encode_special_token_flag = not ("Llama-3.1" in self.args.draft_model and "Llama-3.1" in self.args.target_model)
            input_ids = self.tokenizer.encode(self.preprocess(input_text), add_special_tokens=encode_special_token_flag)
            input_ids = torch.tensor(input_ids, device="cuda:1").unsqueeze(0)

            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            torch.cuda.synchronize()
            start_time = time.time()
            generate_ids = decoding(input_ids)
            torch.cuda.synchronize()
            end_time = time.time()

            if self.accelerator.is_main_process:
                output = self.postprocess(input_text, self.tokenizer.decode(generate_ids[0, :]))
                print("\n================ Generated Output ================")
                print(output)
                print("\n================ Performance Info ================")
                print(f"Time: {end_time - start_time:.2f}s  |  Tokens: {generate_ids.shape[1] - input_ids.shape[1]}  |  Speed: {(generate_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.2f} tokens/s")

if __name__ == "__main__":
    args = parse_arguments()
    alg = ManualPromptEval(args)
    alg.eval()
