import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from zs_fhn.analysis.phase_plane import (
    u_nullcline, v_nullcline, vector_field, find_fixed_points
)
from zs_fhn.models.ode import simulate_ode
import plotly.express as px
import pandas as pd

from ipywidgets import FloatSlider, Button, VBox, IntProgress, HTML
from IPython.display import display, clear_output


def plot_phase_plane(params, u_range=(-2,2), v_range=(-2,2)):
    """
    Plot phase plane with:
    - nullclines
    - vector field
    - fixed points
    """

    # nullclines
    u_vals = np.linspace(u_range[0], u_range[1], 500)
    v_u = u_nullcline(u_vals, params)
    v_v = v_nullcline(u_vals, params)

    # vector field
    U, V = np.meshgrid(
        np.linspace(u_range[0], u_range[1], 20),
        np.linspace(v_range[0], v_range[1], 20)
    )

    dU, dV = vector_field(U, V, params)

    # normalize arrows (optional but clearer)
    mag = np.sqrt(dU**2 + dV**2)
    mag[mag == 0] = 1
    dU /= mag
    dV /= mag

    # plot
    plt.figure(figsize=(6,6))

    plt.quiver(U, V, dU, dV, alpha=0.5)

    plt.plot(u_vals, v_u, label="u-nullcline")
    plt.plot(u_vals, v_v, label="v-nullcline")

    # fixed points
    fps = find_fixed_points(params)
    for (u_fp, v_fp) in fps:
        plt.scatter(u_fp, v_fp, color="red", zorder=5)

    plt.xlabel("u")
    plt.ylabel("v")
    plt.legend()
    plt.title("Phase Plane")

    plt.show()



def plot_time_series(t, u, v):
    plt.figure()

    plt.plot(t, u, label="u(t)")
    plt.plot(t, v, label="v(t)")

    plt.xlabel("t")
    plt.legend()
    plt.title("Time Series")

    plt.show()



def plot_vector_field(U, V, dU, dV):
    """
    Plot vector field (quiver).
    """
    pass


def plot_space_time(u_xt):
    """
    Plot space-time diagram.
    """
    pass


def plot_snapshot(x, u):
    """
    Plot spatial snapshot at fixed time.
    """
    pass


def plot_phase_plane_with_trajectory(u0, v0, params):
    """
    Plot phase plane with:
        - nullclines
        - vector field
        - trajectory from initial condition
        - time series

    Parameters
    ----------
    u0 : float
    v0 : float
    params : dict
    """

    # ===== Run simulation =====
    t, u, v = simulate_ode(u0, v0, params)

    # ===== Create figure =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ==================================================
    # 🟩 LEFT: Phase Plane
    # ==================================================
    ax = axes[0]

    # ----- Nullclines -----
    u_vals = np.linspace(-2, 2, 500)
    v_u = u_nullcline(u_vals, params)
    v_v = v_nullcline(u_vals, params)

    ax.plot(u_vals, v_u, label="u-nullcline", linewidth=2)
    ax.plot(u_vals, v_v, label="v-nullcline", linewidth=2)

    # ----- Vector field -----
    U, V = np.meshgrid(
        np.linspace(-2, 2, 20),
        np.linspace(-2, 2, 20)
    )

    dU, dV = vector_field(U, V, params)

    # 🔑 可选：归一化（让箭头长度一致，更清晰）
    magnitude = np.sqrt(dU**2 + dV**2) + 1e-8
    dU_norm = dU / magnitude
    dV_norm = dV / magnitude

    ax.quiver(U, V, dU_norm, dV_norm, alpha=0.6)

    # ----- Trajectory -----
    ax.plot(u, v, 'r-', linewidth=2, label="trajectory")
    ax.plot(u[0], v[0], 'go', label="start")
    ax.plot(u[-1], v[-1], 'ko', label="end")

    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Phase Plane")
    ax.legend()
    ax.grid()

    # ==================================================
    # 🟦 RIGHT: Time Series
    # ==================================================
    ax = axes[1]

    ax.plot(t, u, label="u(t)")
    ax.plot(t, v, label="v(t)")

    ax.set_xlabel("t")
    ax.set_title("Time Series")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()



def interactive_phase_plane(base_params):
    """
    Interactive phase plane + time series visualization.
    """

    def _plot(u0, v0, a, b, epsilon):

        # ----- update params -----
        params = base_params.copy()
        params["a"] = a
        params["b"] = b
        params["epsilon"] = epsilon

        # ----- simulate -----
        t, u, v = simulate_ode(u0, v0, params)

        # ===== Figure =====
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ==================================================
        # 🟩 Phase Plane
        # ==================================================
        ax = axes[0]

        u_vals = np.linspace(-2, 2, 500)
        v_u = u_nullcline(u_vals, params)
        v_v = v_nullcline(u_vals, params)

        ax.plot(u_vals, v_u, label="u-nullcline")
        ax.plot(u_vals, v_v, label="v-nullcline")

        ax.plot(u, v, 'r-', label="trajectory")
        ax.plot(u[0], v[0], 'go')
        ax.plot(u[-1], v[-1], 'ko')

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title("Phase Plane")
        ax.legend()
        ax.grid()

        # ==================================================
        # 🟦 Time Series
        # ==================================================
        ax = axes[1]

        ax.plot(t, u, label="u(t)")
        ax.plot(t, v, label="v(t)")

        ax.set_xlabel("t")
        ax.set_title("Time Series")
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.show()

    interact(
        _plot,
        u0=FloatSlider(min=-2, max=2, step=0.1, value=0.0),
        v0=FloatSlider(min=-2, max=2, step=0.1, value=0.0),
        a=FloatSlider(min=-1, max=1, step=0.05, value=base_params["a"]),
        b=FloatSlider(min=0.1, max=2, step=0.05, value=base_params["b"]),
        epsilon=FloatSlider(min=0.01, max=0.2, step=0.01, value=base_params["epsilon"])
    )

    def plot_3d_scan(results):
        """
        Plot 3D parameter scan using Plotly.
        """

        import pandas as pd

        df = pd.DataFrame(results, columns=["a", "b", "epsilon", "osc"])

        fig = px.scatter_3d(
            df,
            x="a",
            y="b",
            z="epsilon",
            color="osc",
            title="3D Parameter Scan (Oscillation)"
        )

        fig.show()



def interactive_3d_scan(base_params):

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import multiprocessing
    import time

    from ipywidgets import (
        FloatSlider, Button, VBox, IntProgress,
        HTML, Dropdown, Text, IntText
    )
    from IPython.display import display, clear_output

    from zs_fhn.experiments.parameter_scan import parameter_scan_3d

    # 🔥 PyCharm稳定显示
    pio.renderers.default = "browser"

    # =========================
    # helper
    # =========================
    def parse_range(text):
        start, end, num = map(float, text.split(","))
        return np.linspace(start, end, int(num))

    # =========================
    # CPU 信息
    # =========================
    cpu_count = multiprocessing.cpu_count()

    n_jobs_input = IntText(
        value=max(cpu_count // 2, 1),   # 🔥 改动：默认一半CPU
        description="n_jobs"
    )

    cpu_hint = HTML(value=f"""
    <b>CPU info:</b><br>
    Detected cores: {cpu_count}<br>
    Recommended: {cpu_count//2} ~ {cpu_count}
    """)

    # =========================
    # UI
    # =========================
    u0_slider = FloatSlider(min=-2, max=2, step=0.1, value=0.0, description="u0")
    v0_slider = FloatSlider(min=-2, max=2, step=0.1, value=0.0, description="v0")

    # 🔥 改动：默认参数范围
    a_input = Text(value="-1.2,1.2,50", description="a")
    b_input = Text(value="-1.0,1.0,50", description="b")
    eps_input = Text(value="0.03,0.1,50", description="eps")

    mode_dropdown = Dropdown(
        options=[
            ("Stable only", "stable_only"),
            ("Osc only", "osc_only"),
            ("Both", "both")
        ],
        value="stable_only",
        description="Mode"
    )

    button = Button(description="Run 3D Scan", button_style="success")
    progress = IntProgress(min=0, max=100, value=0)
    status = HTML(value="Idle")

    # =========================
    # main
    # =========================
    def on_click(_):

        clear_output(wait=True)
        display(VBox([
            cpu_hint,
            u0_slider, v0_slider,
            a_input, b_input, eps_input,
            n_jobs_input,
            mode_dropdown,
            button, progress, status
        ]))

        start_time = time.time()   # 🔥 新增：计时开始

        u0 = u0_slider.value
        v0 = v0_slider.value
        mode = mode_dropdown.value

        # =========================
        # 参数解析
        # =========================
        try:
            a_vals = parse_range(a_input.value)
            b_vals = parse_range(b_input.value)
            eps_vals = parse_range(eps_input.value)
        except Exception as e:
            status.value = f"<b style='color:red;'>Input error: {e}</b>"
            return

        total_points = len(a_vals) * len(b_vals) * len(eps_vals)

        # =========================
        # CPU 控制
        # =========================
        n_jobs = n_jobs_input.value

        if n_jobs < 1:
            n_jobs = 1
        if n_jobs > cpu_count:
            n_jobs = cpu_count

        # =========================
        # 进度回调
        # =========================
        def update_progress(i, total):
            pct = int(i / total * 100)
            progress.value = pct
            status.value = f"<b>Scanning: {i}/{total} ({pct}%)</b>"

        status.value = f"<b>Starting scan ({total_points} points)...</b>"
        progress.value = 0

        # =========================
        # scan
        # =========================
        results = parameter_scan_3d(
            a_vals=a_vals,
            b_vals=b_vals,
            eps_vals=eps_vals,
            base_params=base_params,
            u0=u0,
            v0=v0,
            fast_mode=False,
            n_jobs=n_jobs,
            show_progress=True,
            progress_callback=update_progress
        )

        # =========================
        # render
        # =========================
        progress.value = 100
        status.value = "<b>Rendering plot...</b>"

        df = pd.DataFrame(results, columns=["a", "b", "epsilon", "osc"])

        if mode == "stable_only":
            df = df[df["osc"] == False]
        elif mode == "osc_only":
            df = df[df["osc"] == True]

        if len(df) == 0:
            status.value = "<b>No data to plot.</b>"
            return

        df["label"] = df["osc"].map({
            True: "Oscillating",
            False: "Stable"
        })

        fig = px.scatter_3d(
            df,
            x="a",
            y="b",
            z="epsilon",
            color="label",
            color_discrete_map={
                "Oscillating": "red",
                "Stable": "blue"
            },
            title=f"3D Scan ({total_points} pts, {n_jobs} cores)"
        )

        fig.update_traces(marker=dict(size=2, opacity=0.6))
        fig.show()

        # =========================
        # 🔥 总时长输出
        # =========================
        total_time = time.time() - start_time
        status.value = f"<b>Done. Total time: {total_time:.2f} s</b>"

    button.on_click(on_click)

    # =========================
    # 初始显示
    # =========================
    display(VBox([
        cpu_hint,
        u0_slider, v0_slider,
        a_input, b_input, eps_input,
        n_jobs_input,
        mode_dropdown,
        button, progress, status
    ]))


def plot_results_3d(
    results,
    max_points=100000,
    mode="balanced",   # "balanced", "osc_only", "stable_only"
    marker_size=2,
    opacity=0.6
    ):
    """
    Plot 3D parameter scan results.

    Parameters
    ----------
    results : list of tuples
        Expected format:
        [
            (a, b, epsilon, is_osc),
            ...
        ]

        Typically returned by:
        zs_fhn.experiments.parameter_scan.parameter_scan_3d

    max_points : int
        Max number of points to plot (downsampling)

    mode : str
        "balanced" / "osc_only" / "stable_only"

    Notes
    -----
    - Stable → blue
    - Oscillating → red
    """

    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.io as pio

    # 🔥 PyCharm稳定
    pio.renderers.default = "browser"

    print("=== plot_results_3d: INPUT CHECK ===")

    # =========================
    # 输入检查（关键）
    # =========================
    try:
        if not isinstance(results, (list, tuple)):
            raise TypeError("Input 'results' must be a list of tuples.")

        if len(results) == 0:
            raise ValueError("Input 'results' is empty.")

        first = results[0]

        if not (isinstance(first, (list, tuple)) and len(first) == 4):
            raise ValueError(
                "Each element in 'results' must be a tuple: (a, b, epsilon, is_osc)"
            )

        print("Input type: list of tuples ✔")
        print("Detected structure: (a, b, epsilon, is_osc) ✔")
        print("Likely source: parameter_scan_3d ✔")

    except Exception as e:
        print("❌ INPUT ERROR:")
        print(str(e))
        print("\nExpected format:")
        print("[(a, b, epsilon, is_osc), ...]")
        print("Generated by: parameter_scan_3d(...)")
        return

    # =========================
    # 转 DataFrame
    # =========================
    print("\nConverting to DataFrame...")
    df = pd.DataFrame(results, columns=["a", "b", "epsilon", "osc"])

    print(f"Total points: {len(df)}")

    # =========================
    # 模式过滤
    # =========================
    if mode == "osc_only":
        df = df[df["osc"] == True]
        print("Mode: oscillating only")
    elif mode == "stable_only":
        df = df[df["osc"] == False]
        print("Mode: stable only")
    else:
        print("Mode: balanced")

    print(f"After filtering: {len(df)}")

    if len(df) == 0:
        print("⚠️ No data points to plot after filtering.")
        return

    # =========================
    # 下采样
    # =========================
    if len(df) > max_points:
        print(f"Downsampling to {max_points} points...")

        if mode == "balanced":
            df_true = df[df["osc"] == True]
            df_false = df[df["osc"] == False]

            n_true = int(max_points * len(df_true) / len(df))
            n_false = max_points - n_true

            df = pd.concat([
                df_true.sample(min(n_true, len(df_true)), random_state=42),
                df_false.sample(min(n_false, len(df_false)), random_state=42)
            ])
        else:
            df = df.sample(max_points, random_state=42)

    print(f"Plotting {len(df)} points...")

    # =========================
    # 标签（固定颜色）
    # =========================
    df["label"] = df["osc"].map({
        True: "Oscillating",
        False: "Stable"
    })

    # =========================
    # Plot
    # =========================
    try:
        fig = px.scatter_3d(
            df,
            x="a",
            y="b",
            z="epsilon",
            color="label",
            color_discrete_map={
                "Oscillating": "red",
                "Stable": "blue"
            },
            opacity=opacity,
            title="3D Parameter Scan"
        )

        fig.update_traces(marker=dict(size=marker_size))
        fig.show()

        print("✅ Plot rendered successfully.")

    except Exception as e:
        print("❌ PLOT ERROR:")
        print(str(e))
        print("\nPossible causes:")
        print("- Plotly rendering issue")
        print("- Browser not available")
        print("- Too many points")