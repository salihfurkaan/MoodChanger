"""
pipeline.py
Orchestrates the full EEG processing pipeline:

  FakeMuseGenerator → preprocessing → feature extraction → state classification → feedback

Can run in:
  - batch mode   (process N chunks, print summary)
  - stream mode  (continuous, print every chunk)
"""

import time
from typing import Optional

from config             import CHUNK_SIZE, SAMPLE_RATE
from data_generator     import FakeMuseGenerator
from preprocessing      import preprocess
from features           import extract_features
from states             import MentalStateClassifier, StateResult
from feedback           import format_result, session_summary


class EEGPipeline:
    """
    End-to-end EEG analysis pipeline.

    Usage
    -----
    >>> pipeline = EEGPipeline(state="relaxed")
    >>> pipeline.run(duration_sec=10)

    Or step-by-step:
    >>> pipeline = EEGPipeline()
    >>> for result in pipeline.iter_results(n_chunks=20):
    ...     print(result.label)
    """

    def __init__(
        self,
        state:         str   = "relaxed",
        noise_level:   float = 0.3,
        verbose:       bool  = True,
        smooth_feats:  bool  = True,
    ):
        self.generator  = FakeMuseGenerator(state=state, noise_level=noise_level)
        self.classifier = MentalStateClassifier()
        self.verbose    = verbose
        self.smooth     = smooth_feats
        self._history   = []
        self._chunk_count = 0

    # ------------------------------------------------------------------
    # Single-chunk processing
    # ------------------------------------------------------------------

    def process_chunk(self) -> StateResult:
        """Pull one chunk through the entire pipeline and return a StateResult."""
        raw       = self.generator.get_chunk()
        processed = preprocess(raw)
        features  = extract_features(processed, smooth=self.smooth)
        result    = self.classifier.classify(features)
        self._history.append(result)
        self._chunk_count += 1
        return result

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_results(self, n_chunks: int):
        """Yield StateResult for each of `n_chunks` chunks."""
        for _ in range(n_chunks):
            yield self.process_chunk()

    # ------------------------------------------------------------------
    # High-level runners
    # ------------------------------------------------------------------

    def run(
        self,
        duration_sec: float = 10.0,
        realtime:     bool  = False,
        print_every:  int   = 4,
    ):
        """
        Run the pipeline for `duration_sec` seconds.

        Args:
            duration_sec: total simulated duration to process
            realtime    : sleep between chunks (True = wall-clock paced)
            print_every : print a result every N chunks (0 = silent)
        """
        n_chunks      = max(1, int(duration_sec * SAMPLE_RATE / CHUNK_SIZE))
        chunk_dur     = CHUNK_SIZE / SAMPLE_RATE
        self._history = []
        self.classifier.reset()

        print(f"\n{'='*55}")
        print(f"  EEG Pipeline  |  state='{self.generator.state}'  "
              f"|  {duration_sec:.1f}s  |  {n_chunks} chunks")
        print(f"{'='*55}\n")

        for i, result in enumerate(self.iter_results(n_chunks)):
            if print_every and (i % print_every == 0 or i == n_chunks - 1):
                if self.verbose:
                    print(format_result(result, verbose=(i == n_chunks - 1)))
                else:
                    emoji = {"relaxed":"😌","focused":"🧠","anxious":"😰",
                             "drowsy":"😴","meditative":"🧘","neutral":"😐"}.get(result.label,"🔬")
                    print(f"  chunk {i+1:03d}/{n_chunks}  {emoji} {result.label:12s}  "
                          f"conf={result.confidence:.0%}")
            if realtime:
                time.sleep(chunk_dur)

        print()
        print(session_summary(self._history))

    def set_state(self, state: str):
        """Dynamically switch the simulated mental state."""
        self.generator.set_state(state)
        self.classifier.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self):
        return list(self._history)

    @property
    def last_result(self) -> Optional[StateResult]:
        return self._history[-1] if self._history else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Pipeline demo")
    parser.add_argument("--state",    default="relaxed",
                        choices=["relaxed","focused","anxious","drowsy","meditative"],
                        help="Simulated mental state")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulated duration in seconds")
    parser.add_argument("--realtime", action="store_true",
                        help="Pace output to wall-clock time")
    parser.add_argument("--quiet",    action="store_true",
                        help="Minimal output (no band-power breakdown)")
    args = parser.parse_args()

    pipeline = EEGPipeline(state=args.state, verbose=not args.quiet)
    pipeline.run(duration_sec=args.duration, realtime=args.realtime)