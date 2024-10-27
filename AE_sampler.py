import torch
from tqdm import tqdm
from typing import Any, Tuple, Optional

class AbsorbEscapeSampler:
    def __init__(self, ar_model: Any, device: str = 'cuda', temperature: float = 1.0):
        """
        Initialize the Absorb-Escape sampler with a more flexible interface.

        Args:
            ar_model: Autoregressive model with a generate() method
            device: Device to run computations on
            temperature: Temperature for sampling
        """
        self.ar_model = ar_model
        self.device = device
        self.temperature = temperature

    def sample(
        self,
        seq: torch.Tensor,
        signal: torch.Tensor,
        diffusion_logits: torch.Tensor,
        diffusion_seq: torch.Tensor,
        absorb_threshold: float = 0.85
    ) -> torch.Tensor:
        """
        Sample using the absorb-escape algorithm.

        Args:
            seq: Original sequence (B x L)
            signal: Conditioning signal (B x ...)
            diffusion_logits: Logits from diffusion model (B x L x V)
            diffusion_seq: Sequence predictions from diffusion model (B x L)
            absorb_threshold: Confidence threshold for absorbing diffusion predictions

        Returns:
            torch.Tensor: Final refined sequences
        """
        # Calculate confidence scores from diffusion model
        normalized_logits = torch.nn.functional.softmax(diffusion_logits, dim=-1)
        confidence_scores, _ = torch.max(normalized_logits, dim=-1)  # B x L

        # Process each sequence independently
        batch_size, seq_len = seq.shape
        final_sequences = torch.zeros_like(diffusion_seq, device=self.device)

        for batch_idx in tqdm(range(batch_size), desc="Processing sequences"):
            final_sequences[batch_idx] = self._process_sequence(
                signal=signal[batch_idx],
                confidence_scores=confidence_scores[batch_idx],
                initial_seq=diffusion_seq[batch_idx].clone(),
                absorb_threshold=absorb_threshold,
                seq_len=seq_len
            )

        return final_sequences

    def _process_sequence(
        self,
        signal: torch.Tensor,
        confidence_scores: torch.Tensor,
        initial_seq: torch.Tensor,
        absorb_threshold: float,
        seq_len: int
    ) -> torch.Tensor:
        """
        Process a single sequence using the absorb-escape mechanism.

        Args:
            signal: Single conditioning signal
            confidence_scores: Confidence scores from diffusion model
            initial_seq: Initial sequence from diffusion model
            absorb_threshold: Confidence threshold
            seq_len: Length of sequence
        """
        position = 0
        result_seq = initial_seq.clone()

        while position < seq_len:
            # Check if we need to escape
            if confidence_scores[position] < absorb_threshold:
                # Get the prefix up to current position
                prefix = result_seq[:position].clone()

                # Generate new sequence from this point
                escape_length, new_sequence, is_complete = self._generate_continuation(
                    prefix=prefix,
                    confidence_scores=confidence_scores,
                    signal=signal,
                    seq_len=seq_len
                )

                if escape_length > position:
                    result_seq[:escape_length] = new_sequence
                    position = escape_length

                if is_complete:
                    break

            position += 1

        return result_seq

    def _generate_continuation(
        self,
        prefix: torch.Tensor,
        confidence_scores: torch.Tensor,
        signal: torch.Tensor,
        seq_len: int
    ) -> Tuple[int, torch.Tensor, bool]:
        """
        Generate continuation using the AR model until diffusion confidence exceeds AR confidence.

        Args:
            prefix: Previous tokens to condition on
            confidence_scores: Diffusion model confidence scores
            signal: Conditioning signal
            seq_len: Maximum sequence length

        Returns:
            tuple: (length generated, generated sequence, whether generation completed)
        """
        # Keep track of where we are
        current_length = len(prefix)
        is_complete = True

        # Generate tokens until we either hit a high confidence diffusion token
        # or reach the end of the sequence
        while current_length < seq_len:
            # Get AR model generation and confidence
            generation_output = self.ar_model.generate(
                prefix=prefix,
                signal=signal.unsqueeze(0),  # Add batch dimension
                max_length=seq_len,
                temperature=self.temperature
            )

            # Unpack AR model outputs - exact format will depend on AR model implementation
            ar_sequence = generation_output.sequence
            ar_confidence = generation_output.confidence

            # Find first position where diffusion model is more confident
            for pos in range(current_length, len(ar_sequence)):
                if confidence_scores[pos] > ar_confidence[pos]:
                    is_complete = False
                    current_length = pos
                    return current_length, ar_sequence[:current_length], is_complete

            # If we made it here, we used the full AR generation
            current_length = len(ar_sequence)
            return current_length, ar_sequence, is_complete

        return current_length, prefix, is_complete

class GenerationOutput:
    """
    Example expected output format from AR model's generate method.
    AR model should return something with these attributes, but exact implementation may vary.
    """
    def __init__(self, sequence: torch.Tensor, confidence: torch.Tensor):
        self.sequence = sequence
        self.confidence = confidence
