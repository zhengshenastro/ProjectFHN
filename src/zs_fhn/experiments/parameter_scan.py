import numpy as np
from zs_fhn.models.ode import simulate_ode
from zs_fhn.analysis.metrics import detect_oscillation
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def _single_eval(args):
    a, b, eps, base_params, u0, v0, fast_mode = args

    params = base_params.copy()
    params["a"] = a
    params["b"] = b
    params["epsilon"] = eps

    if fast_mode:
        params["T"] = 120   # 稍微保守一点

    t, u, v = simulate_ode(u0, v0, params)

    is_osc = detect_oscillation(u, t)

    return (a, b, eps, is_osc)


def parameter_scan_3d(
    a_vals,
    b_vals,
    eps_vals,
    base_params,
    u0=0.0,
    v0=0.0,
    fast_mode=False,
    n_jobs=None,
    show_progress=True,
    progress_callback=None   # 🔥 新增
):
    import numpy as np
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    tasks = [
        (a, b, eps, base_params, u0, v0, fast_mode)
        for a in a_vals
        for b in b_vals
        for eps in eps_vals
    ]

    total = len(tasks)
    results = []

    with Pool(n_jobs) as pool:

        iterator = pool.imap_unordered(_single_eval, tasks)

        if show_progress:
            iterator = tqdm(iterator, total=total)

        for i, res in enumerate(iterator, 1):
            results.append(res)

            # 🔥 同步 UI 进度
            if progress_callback is not None:
                progress_callback(i, total)

    return results