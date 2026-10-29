"""CPU contracts; these tests do not establish TTC accuracy on real data."""

import unittest

import numpy as np
import torch

from e_jepa_ttc.range_transfer.mass import fit_mass_plans
from e_jepa_ttc.range_transfer.objective import range_transfer_loss
from e_jepa_ttc.simplex_t.model import training_loss
from e_jepa_ttc.simplex_t.phase import emitted_phase, ttc_to_phase


class MassTests(unittest.TestCase):
    def example(self):
        t = np.array(
            [0.7] + [1.5] * 8 + [2.5] * 16 + [4.5] * 25 + [7.0] * 8 + [9.0] * 4 + [-3.0] * 18
        )
        g = np.array(["a"] * len(t))
        m = np.ones(len(t)) / len(t)
        return t, g, m

    def test_native_unchanged(self):
        t, g, m = self.example()
        m = m.astype("float32")
        p = fit_mass_plans(t, g, m, fit_role="H")
        np.testing.assert_array_equal(p.native, m)
        self.assertEqual(p.native.dtype, m.dtype)

    def test_benchmark_masses(self):
        t, g, m = self.example()
        p = fit_mass_plans(t, g, m, fit_role="H")
        for mask, w in [
            (t <= 0, 0.1),
            ((t > 0) & (t <= 3), 0.5),
            ((t > 3) & (t <= 6), 0.3),
            (t > 6, 0.1),
        ]:
            self.assertAlmostEqual(p.benchmark[mask].sum(), w)

    def test_subrange_retains_outer_mass(self):
        t, g, m = self.example()
        p = fit_mass_plans(t, g, m, fit_role="H")
        for lo, hi in [(-10, 0), (0.1, 3), (3, 6), (6, 10)]:
            mask = (t > lo) & (t <= hi)
            self.assertAlmostEqual(p.benchmark[mask].sum(), p.subrange[mask].sum())

    def test_subrange_cap(self):
        t, g, m = self.example()
        p = fit_mass_plans(t, g, m, fit_role="H")
        ratio = p.subrange / p.benchmark
        self.assertTrue((ratio >= 0.25 - 1e-12).all())
        self.assertTrue((ratio <= 4 + 1e-12).all())
        self.assertGreater(p.subrange[0], p.benchmark[0])

    def test_preserves_group_totals(self):
        t, g, m = self.example()
        t = np.r_[t, [-2.0, 1.2, 4.0, 9.0]]
        g = np.r_[g, ["b"] * 4]
        m = np.r_[m * 0.7, np.ones(4) * 0.3 / 4]
        p = fit_mass_plans(t, g, m, fit_role="H")
        for weights in [p.native, p.benchmark, p.subrange]:
            self.assertAlmostEqual(weights[g == "a"].sum(), 0.7)
            self.assertAlmostEqual(weights[g == "b"].sum(), 0.3)

    def test_absent_bands_training_only(self):
        t = np.array([1.5, 2.5])
        m = np.array([0.5, 0.5])
        p = fit_mass_plans(t, np.array(["a", "a"]), m, fit_role="H")
        self.assertAlmostEqual(p.benchmark.sum(), 1)

    def test_rejects_evaluation_roles(self):
        t, g, m = self.example()
        for role in ["V", "DEV32", "FCWD", "TRAIN40", "P"]:
            with self.subTest(role=role), self.assertRaises(ValueError):
                fit_mass_plans(t, g, m, fit_role=role)

    def test_refuses_outside_support(self):
        for t in [
            [1.0, 11.0],
            [0.0, 2.0],
            [0.1, 2.0],
            [-11.0, 1.0],
            [-10.0, 1.0],
            [float("nan"), 1.0],
        ]:
            with self.subTest(t=t), self.assertRaises(ValueError):
                fit_mass_plans(
                    np.array(t), np.array(["a", "a"]), np.array([0.5, 0.5]), fit_role="H"
                )

    def test_refuses_bad_mass(self):
        for m in [[0.5, 0.4], [0.5, -0.5], [float("nan"), 0.5]]:
            with self.subTest(m=m), self.assertRaises(ValueError):
                fit_mass_plans(
                    np.array([1.0, 2.0]), np.array(["a", "a"]), np.array(m), fit_role="H"
                )

    def test_row_permutation_equivariant(self):
        t, g, m = self.example()
        order = np.random.default_rng(1).permutation(len(t))
        a = fit_mass_plans(t, g, m, fit_role="H")
        b = fit_mass_plans(t[order], g[order], m[order], fit_role="H")
        np.testing.assert_allclose(a.subrange[order], b.subrange, rtol=1e-13)

    def test_single_subrange(self):
        t = np.array([1.2, 1.3, -2.0])
        p = fit_mass_plans(t, np.array(["a"] * 3), np.ones(3) / 3, fit_role="H")
        np.testing.assert_allclose(p.benchmark, p.subrange)


class ObjectiveTests(unittest.TestCase):
    def inputs(self):
        truth = ttc_to_phase(torch.tensor([1.2, 2.0, 4.0, 9.0, -2.0, -8.0], dtype=torch.float64))
        raw = (
            (truth + torch.tensor([-0.02, 0.01, -0.002, 0.001, 0.003, -0.001], dtype=torch.float64))
            .detach()
            .requires_grad_()
        )
        experts = torch.stack([truth * 0.9, truth * 1.2, truth * 1.1], 1)
        output = {
            "raw_location": raw,
            "point_phase": emitted_phase(raw),
            "q10": raw - 0.01,
            "q90": raw + 0.01,
            "relative_cost": raw[:, None] * torch.ones(1, 3, dtype=raw.dtype),
        }
        return (
            output,
            truth,
            experts,
            torch.tensor([0.1, 0.1, 0.3, 0.1, 0.2, 0.2], dtype=torch.float64),
            6,
        )

    def test_zero_is_exact_native_loss_and_gradient(self):
        args = self.inputs()
        a = training_loss(*args)
        b = range_transfer_loss(*args, relative_weight=0)
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a, args[0]["raw_location"], retain_graph=True)[0]
        gb = torch.autograd.grad(b, args[0]["raw_location"])[0]
        self.assertTrue(torch.equal(ga, gb))

    def test_added_term_finite_with_negative_targets(self):
        args = self.inputs()
        v = range_transfer_loss(*args, relative_weight=0.25)
        self.assertTrue(torch.isfinite(v))
        v.backward()
        self.assertTrue(torch.isfinite(args[0]["raw_location"].grad).all())

    def test_exact_predictions_have_no_extra_penalty(self):
        o, t, e, m, n = self.inputs()
        o["raw_location"] = t
        o["point_phase"] = emitted_phase(t)
        a = training_loss(o, t, e, m, n)
        b = range_transfer_loss(o, t, e, m, n, relative_weight=0.25)
        self.assertLess(abs(float((a - b).detach())), 1e-14)

    def test_added_term_nonnegative(self):
        args = self.inputs()
        self.assertGreaterEqual(
            float(range_transfer_loss(*args, relative_weight=0.25).detach()),
            float(training_loss(*args).detach()),
        )

    def test_nonfinite_coeff_rejected(self):
        args = self.inputs()
        for w in [-0.1, float("nan"), float("inf")]:
            with self.subTest(w=w), self.assertRaises(ValueError):
                range_transfer_loss(*args, relative_weight=w)

    def test_batch_mass_not_normalized(self):
        args = list(self.inputs())
        a = range_transfer_loss(*args, relative_weight=0.25)
        args[3] = args[3] * 0.2
        b = range_transfer_loss(*args, relative_weight=0.25)
        self.assertAlmostEqual(float(b.detach()), 0.2 * float(a.detach()), places=12)

    def test_accumulation_value_equivalence(self):
        o, t, e, m, n = self.inputs()
        whole = range_transfer_loss(o, t, e, m, n, relative_weight=0.25)
        parts = []
        for sl in [slice(0, 2), slice(2, 6)]:
            part = {k: v[sl] for k, v in o.items()}
            parts.append(
                range_transfer_loss(part, t[sl], e[sl], m[sl], n, relative_weight=0.25)
                * len(t[sl])
                / len(t)
            )
        self.assertAlmostEqual(float((whole - sum(parts)).detach()), 0, places=12)

    def test_accumulation_gradient_equivalence(self):
        o, t, e, m, n = self.inputs()
        whole = range_transfer_loss(o, t, e, m, n, relative_weight=0.25)
        gw = torch.autograd.grad(whole, o["raw_location"], retain_graph=True)[0]
        pieces = []
        for sl in [slice(0, 2), slice(2, 6)]:
            part = {k: v[sl] for k, v in o.items()}
            pieces.append(
                range_transfer_loss(part, t[sl], e[sl], m[sl], n, relative_weight=0.25)
                * len(t[sl])
                / len(t)
            )
        gp = torch.autograd.grad(sum(pieces), o["raw_location"])[0]
        self.assertTrue(torch.allclose(gw, gp, atol=1e-14, rtol=1e-14))

    def test_decoder_dead_zone_is_not_secretly_straight_through(self):
        o, t, e, m, n = self.inputs()
        raw = torch.zeros_like(t, requires_grad=True)
        o = {
            **o,
            "raw_location": raw,
            "point_phase": emitted_phase(raw),
            "q10": raw - 0.01,
            "q90": raw + 0.01,
            "relative_cost": raw[:, None] * torch.ones(1, 3),
        }
        extra = range_transfer_loss(o, t, e, m, n, relative_weight=0.25) - training_loss(
            o, t, e, m, n
        )
        grad = torch.autograd.grad(extra, raw)[0]
        self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_no_output_mutation(self):
        args = self.inputs()
        before = {k: v.detach().clone() for k, v in args[0].items()}
        range_transfer_loss(*args, relative_weight=0.25)
        for k, v in args[0].items():
            self.assertTrue(torch.equal(v.detach(), before[k]))


if __name__ == "__main__":
    unittest.main()
