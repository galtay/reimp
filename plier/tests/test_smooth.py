import numpy as np
import pytest

from reimp_plier.model import num_pc
from reimp_plier.smooth import smooth

# Every expected value below was computed by R 4.5.3: smooth(x, twiceit = TRUE).


@pytest.mark.parametrize(
    "x, expected",
    [
        # R's ?smooth example.
        ([4, 1, 3, 6, 6, 4, 1, 6, 2, 4, 2], [3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2]),
        # Two-point plateaus at peaks and valleys, which the "S" step splits.
        (
            [5, 5, 3, 3, 8, 8, 2, 9, 9, 1, 7, 7, 4, 6, 6, 0, 2, 2, 5, 3],
            [5, 5, 5, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 2, 2, 2, 3, 3],
        ),
    ],
)
def test_smooth_matches_r(x, expected) -> None:
    np.testing.assert_array_equal(smooth(np.array(x, dtype=float)), expected)


def _singular_values() -> np.ndarray:
    i = np.arange(1, 61)
    return 100 / i**0.7 + (i % 7) / 10


# smooth(abs(diff(diff(d))), twiceit = TRUE) for d = _singular_values(), in R.
R_SMOOTHED_SECOND_DIFFERENCES = [
    14.3253518557707693, 6.7575234756118121, 2.9736092855323335, 1.5961479518778638,
    1.3327520875058276, 0.4394361012728289, 0.4394361012728289, 0.3188469396253666,
    0.2394306876718808, 0.2394306876718808, 0.1848360019440491, 0.1848360019440491,
    0.1053121385200892, 0.0797463595030177, 0.0669634699944819, 0.0568313510600760,
    0.0568313510600760, 0.0486887627271670, 0.0486887627271670, 0.0347060185091106,
    0.0282978515811347, 0.0250937681171468, 0.0223668497379119, 0.0223668497379119,
    0.0200303187275210, 0.0200303187275210, 0.0157602078315797, 0.0134122770558953,
    0.0122383116680531, 0.0112007421013676, 0.0112007421013676, 0.0102800507991319,
    0.0102800507991319, 0.0085454147631694, 0.0074784102486340, 0.0069449079913664,
    0.0064622196303485, 0.0064622196303485, 0.0060243514863032, 0.0060243514863032,
    0.0051850572075463, 0.0046278334408036, 0.0043492215574323, 0.0040930943862190,
    0.0040930943862190, 0.0038571961026523, 0.0038571961026523, 0.0034001711272413,
    0.0030794460316592, 0.0029190834838682, 0.0027699483212560, 0.0027699483212560,
    0.0026310605730826, 0.0026310605730826, 0.0023600641652504, 0.0021617036101560,
    0.0020625233326088, 0.0019694747084262,
]  # fmt: skip


def test_smooth_of_second_differences_matches_r() -> None:
    smoothed = smooth(np.abs(np.diff(_singular_values(), n=2)))
    np.testing.assert_allclose(smoothed, R_SMOOTHED_SECOND_DIFFERENCES, rtol=1e-12)


def test_num_pc_matches_r() -> None:
    # R: x <- smooth(abs(diff(diff(d))), twiceit = TRUE); which(x <= quantile(x, 0.5))[1] + 1
    assert num_pc(_singular_values()) == 31
