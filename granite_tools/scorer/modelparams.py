"""For fitting the cubic spline model (bigram/unigram scores smoothing and interpolation)"""

SPLINE_BSPLINE_DEGREE = 3
SPLINE_KNOT_SEGMENTS = 35
SPLINE_LAMBDA_SMOOTHING = 1
SPLINE_KAPPA_PENALTY = 1e6

SPLINE_KWARGS = {
    "bspline_degree": SPLINE_BSPLINE_DEGREE,
    "knot_segments": SPLINE_KNOT_SEGMENTS,
    "lambda_smoothing": SPLINE_LAMBDA_SMOOTHING,
    "kappa_penalty": SPLINE_KAPPA_PENALTY,
}
