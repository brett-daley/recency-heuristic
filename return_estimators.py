import numpy as np


def calculate(estimator: str, v, next_v, rewards, terminateds, truncateds, discount):
    dones = np.logical_or(terminateds, truncateds)  # End of episode, regardless of reason
    bootstraps = np.where(terminateds, 0.0, next_v)
    td_errors = rewards + (discount * bootstraps) - v

    tokens = estimator.split('-')  # Estimator names are strings delimited by '-'

    if tokens[0] == 'lambda':  # Uncorrected lambda-returns
        assert len(tokens) == 2
        lambd = float(tokens[1])
        td_lambda_errors = _calculate_td_lambda_errors(lambd, td_errors, discount, dones)
        return td_lambda_errors + v

    elif tokens[0] == 'nstep':  # n-step returns
        assert len(tokens) == 2
        n = int(tokens[1])
        return _calculate_n_step_returns(n, rewards, bootstraps, discount, dones)

    elif tokens[0] == 'trunc':
        assert len(tokens) == 3
        lambd = float(tokens[1])
        L = int(tokens[2])

        G = 0.0
        total_weight = 0.0

        for i in range(L - 1):
            n = i + 1
            w_n = (1-lambd) * pow(lambd, n - 1)
            total_weight += w_n
            G += w_n * _calculate_n_step_returns(n, rewards, bootstraps, discount, dones)

        G += pow(lambd, L - 1) * _calculate_n_step_returns(L, rewards, bootstraps, discount, dones)
        return G

    elif tokens[0] == 'sparse':
        assert len(tokens) == 3
        lambd = float(tokens[1])
        m = int(tokens[2])
        L = 10  # Truncates calculation for feasibility

        G = 0.0
        total_weight = 0.0
        n_func = lambda k: m * (k-1) + 1

        for k in range(1, L):
            weight = (1-lambd) * pow(lambd, k - 1)
            total_weight += weight
            G += weight * _calculate_n_step_returns(n_func(k), rewards, bootstraps, discount, dones)

        assert total_weight <= 1.0
        G += (1 - total_weight) * _calculate_n_step_returns(n_func(L), rewards, bootstraps, discount, dones)
        return G

    else:
        raise ValueError(f"unknown return estimator '{estimator}'")


def _calculate_n_step_returns(n, rewards, bootstraps, discount, should_cut):
    assert n >= 1
    T = len(rewards)

    one_step_returns = rewards + (discount * bootstraps)
    if n == 1:
        return one_step_returns

    n_step_returns = np.copy(one_step_returns)
    for t in range(T):
        start = t
        end = min(t + n - 1, T - 1)

        G = one_step_returns[end]
        for i in reversed(range(start, end)):
            G = np.where(should_cut[i], bootstraps[i], G)  # Cut for off-policy corrections
            G = rewards[i] + (discount * G)

        n_step_returns[t] = G

    return n_step_returns


def _calculate_td_lambda_errors(lambd, td_errors, discount, dones):
    assert 0.0 <= lambd <= 1.0
    T = len(td_errors)

    td_lambda_errors = np.copy(td_errors)
    weights = np.where(dones[:-1], 0.0, discount * lambd)  # Cut for off-policy corrections

    E = td_lambda_errors[-1]
    for t in reversed(range(T - 1)):  # For each timestep in trajectory backwards
        E = td_errors[t] + (weights[t] * E)
        td_lambda_errors[t] = E

    return td_lambda_errors
