import numpy as np
from hmmlearn.hmm import GaussianHMM


def fit_hmm(features, n_regimes=2, n_restarts=5, random_state=42):
    """
    Fit a Gaussian HMM with multiple random restarts, keep the best.
    Diagonal covariance keeps parameter count low.
    """
    best_model = None
    best_score = -np.inf
    rng = np.random.RandomState(random_state)

    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=300,
            random_state=rng.randint(0, 10000),
            tol=1e-4,
            init_params="stmc",
            verbose=False,
        )
        try:
            model.fit(features)
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=300,
            random_state=random_state,
        )
        model.fit(features)
        return model

    return best_model


def get_regime_probabilities(model, features):
    """Return filtered state probabilities for the last observation."""
    return model.predict_proba(features)[-1]


def label_regimes(model):
    """
    Assign labels based on the HMM's learned means.
    For 2 regimes: higher eq_return mean → 'risk_on', lower → 'risk_off'.
    For 3 regimes: growth / transition / crisis.
    """
    eq_return_means = model.means_[:, 0]
    n = model.n_components
    order = np.argsort(eq_return_means)

    if n == 2:
        return {order[0]: "risk_off", order[1]: "risk_on"}

    labels = {}
    for rank, idx in enumerate(order):
        if rank == 0:
            labels[idx] = "crisis"
        elif rank == n - 1:
            labels[idx] = "growth"
        else:
            labels[idx] = "transition"
    return labels


def predict_states(model, features):
    """Viterbi decoding."""
    return model.predict(features)
