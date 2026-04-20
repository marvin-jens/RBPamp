import numpy as np
import shelve
import os
import logging


class PSAMErrorEstimator(object):
    def __init__(
        self, descentpath, q=[25.0, 50.0, 75.0], tol=0.05, use_shelve=None, max_t=None
    ):
        self.q = np.array(q)
        self.descentpath = descentpath
        self.tol = tol
        self.logger = logging.getLogger("opt.PSAMErrorEstimator")
        if use_shelve:
            self.shelve = use_shelve
        else:
            try:
                spath = os.path.join(descentpath, "history")
                self.shelve = shelve.open(spath, flag="r")
            except IOError:
                self.logger.error(
                    "no history to estimate error from in '{}'".format(descentpath)
                )
                self.shelve = None

        if max_t is None:
            self.max_t = self.find_max_t()
        else:
            self.max_t = max_t

        self.logger.debug(f"{self.descentpath} max_t={self.max_t}")

    def find_max_t(self):
        if self.shelve is None:
            return 0

        t = -1
        while "params_t{}".format(t + 1) in self.shelve:
            t += 1
        return t

    def get(self, name, t):
        if t == -1:
            t = self.max_t

        key = "{0}_t{1}".format(name, t)
        if key in self.shelve:
            return self.shelve[key]

    def load_data(self):
        params = []
        stats = []
        t = 0
        while True:
            par = self.get("params", t)
            stat = self.get("stats", t)
            if par is None or stat is None:
                break

            if t == 0:
                ref_par = par
                ref_pd = par.get_data()

            pd = par.get_data()
            if pd.shape == ref_pd.shape:
                params.append(pd)
                stats.append(stat)
            else:
                self.logger.warning(
                    f"ignoring t={t} with PSAM data shape "
                    f"{pd.shape}, differing from {ref_pd.shape}"
                    f" at t0 (presumably from a previous run?)."
                )
            t += 1

        if not params:
            raise ValueError("could not load any error estimation data!")

        return ref_par, np.array(params), np.array(stats)

    def estimate(self, save=True, t_ref=-1):
        if self.shelve is None:
            return None

        if t_ref == -1:
            t_ref = self.max_t

        params0 = self.get("params", 0)
        pd0 = params0.get_data()
        params, stats = self.load_data()
        mdl_errors = np.array([s.error for s in stats])

        cut = mdl_errors[t_ref] * (1 + self.tol)

        I = (mdl_errors < cut)[:t_ref].nonzero()[0]
        if len(I) < 10:
            self.logger.warning(
                "can not estimate errors for t_ref={0}. Insufficient data n={1}.".format(
                    t_ref, len(I)
                )
            )
            return

        self.logger.debug(
            "esimating errors from n={0} data points for reference t={1}".format(
                len(I), t_ref
            )
        )
        param_data = [par.get_data() for par in params[I]]
        param_data = np.array([pd for pd in param_data if pd.shape == pd0.shape])

        # print(param_data.shape, param_data.dtype)
        params_q = np.percentile(param_data, self.q, axis=0)

        p_q = [params0.copy().set_data(perc) for perc in params_q]
        if save:
            for p, q in zip(p_q, self.q):
                p.save(os.path.join(self.descentpath, "parameters_q{}.tsv".format(q)))

        p_lo, p_mid, p_hi = p_q
        p_mid.lo = p_lo
        p_mid.hi = p_hi

        return p_mid


if __name__ == "__main__":
    from collections import defaultdict
    from RBPamp import dominguez_rbps
    import pandas as pd

    data = defaultdict(list)
    for rbp in dominguez_rbps:
        err = PSAMErrorEstimator(
            f"/home/mjens/engaging/RBNS/{rbp}/RBPamp/py3_seed1/opt_nostruct/"
        )
        mdl, params, stats = err.load_data()
        print(f"received sub-sampled parameters shape={params.shape}")
        mdl_error = np.array([s.error for s in stats])
        # mdl_corr = np.array([s.correlations for s in stats])
        # mdl_errors = np.array([s.sample_errors for s in stats])

        e0 = mdl_error.min()
        i_opt = mdl_error.argmin()
        mdl_opt = params[i_opt]
        eps = 0.05
        to_plot = []
        psam_std = []
        eps_choices = [0.01, 0.05, 0.1, 0.25]
        for eps in eps_choices:
            I = mdl_error <= (1 + eps) * e0
            print(
                f"eps={eps} selectes model error range from {mdl_error[I].min()} to {mdl_error[I].max()} n={I.sum()}"
            )
            Kd = 1.0 / params[I, 0]
            eps_err = mdl_error[I] / e0
            to_plot.append(Kd)
            psam_std = params[I, :].std(axis=0).max()
            for i, kd in enumerate(Kd):
                data["rbp"].append(rbp)
                data["eps"].append(eps)
                data["eps_sample"].append(i)
                data["eps_error"].append(eps_err[i])
                data["Kd"].append(kd)
                data["PSAM_max_stddev"].append(psam_std)

    df = pd.DataFrame(data)
    df.to_csv(
        "/home/mjens/git/RBPamp/rRBNS/eps_err_nostruct_seed1_std.tsv",
        sep="\t",
        index=False,
    )

    1 / 0
    for rbp in dominguez_rbps:
        err = PSAMErrorEstimator(
            f"/home/mjens/engaging/RBNS/{rbp}/RBPamp/py3/opt_struct/"
        )
        mdl, params, stats = err.load_data()
        print(f"received sub-sampled parameters shape={params.shape}")
        mdl_error = np.array([s.error for s in stats])
        e0 = mdl_error.min()
        eps = 0.05
        to_plot = []
        eps_choices = [0.01, 0.05, 0.1, 0.25]
        for eps in eps_choices:
            I = mdl_error <= (1 + eps) * e0
            print(
                f"eps={eps} selectes model error range from {mdl_error[I].min()} to {mdl_error[I].max()} n={I.sum()}"
            )
            Kd = 1.0 / params[I, 0]
            to_plot.append(Kd)

        import RBPamp.report  # configure matplotlib style and backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.violinplot(to_plot, showmeans=True)
        ax.set_ylabel("$K_d^{opt}$ estimate [nM]")
        ax.set_xticks(range(1, len(to_plot) + 1))
        ax.set_xticklabels(eps_choices)
        ax.set_xlabel("model error tolerance")
        fig.tight_layout()
        fig.savefig(f"Kd_violin_{rbp}.pdf")
        plt.close()
    # print(params[I].var(axis=0))